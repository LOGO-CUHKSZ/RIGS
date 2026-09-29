"""Portable, streaming access to the supplied USPTO data. No experiment runs."""
from pathlib import Path
import argparse
import csv
import gzip
import itertools
import json


class DataPackage:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.catalog = json.loads((self.root / 'catalog.json').read_text())

    def dataset(self, name):
        entry = self.catalog['datasets'][name]
        return self.root / entry['path']

    def resolve(self, component, relative):
        target = (component / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError('A dataset reference leaves the package: ' + str(relative))
        if not target.is_file():
            raise FileNotFoundError(target)
        return target

    def records_path(self, name, split='train', kind='groups'):
        component = self.dataset(name)
        if component.is_file():
            return component
        manifest_path = component / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        canonical = 'validation' if split == 'val' else split
        cache = 'val' if split == 'validation' else split
        artifact_key = {'groups':'semantic_groups','cases':'cases','evaluation':'evaluation_mappings',
                        'metrics':'route_metric_values','shuffle':'shuffle_donor_pools'}.get(kind,kind)
        artifacts = manifest.get('artifacts',{})
        if isinstance(artifacts,dict) and f'{artifact_key}.{canonical}' in artifacts:
            return self.resolve(component, artifacts[f'{artifact_key}.{canonical}'])
        output = manifest.get('outputs',{}).get(kind,{})
        if cache in output:
            return self.resolve(component,output[cache]['path'])
        candidates = []
        if kind == 'groups':
            candidates = [f'groups/{cache}.jsonl',f'groups/{canonical}.jsonl.gz']
        elif kind == 'routes':
            candidates = [f'endpoint_graphs/routes_{cache}.jsonl',f'route_records_{cache}.jsonl']
        elif kind == 'raw':
            candidates = [f'raw/raw_{cache}.csv']
        elif kind == 'evaluation':
            candidates = [f'strict100_pista_frozen_{canonical}_m100.jsonl.gz',
                          f'frozen_{canonical}_axis4_rc5m100s50.jsonl']
        elif kind == 'cases':
            candidates = [f'cases/{canonical}.jsonl.gz']
        for rel in candidates:
            if (component/rel).is_file():
                return self.resolve(component,rel)
        raise KeyError(f'{name} does not provide {kind}/{split}; consult catalog.json')

    def rows(self, name, split='train', kind='groups'):
        path = self.records_path(name,split,kind)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path,'rt',encoding='utf-8') as stream:
            if path.suffix == '.csv':
                yield from csv.DictReader(stream)
            elif path.suffix == '.json':
                yield json.load(stream)
            else:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    def graph(self, name, split='train', graph_index=0):
        """Read one indexed endpoint graph on CPU; requires PyTorch and PyG.

        This convenience reader opens a shard on each call. A training loader
        should maintain its own bounded shard cache.
        """
        import torch
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        from torch_geometric.data.separate import separate

        if graph_index < 0:
            raise IndexError('graph_index must be nonnegative')
        component = self.dataset(name)
        cache = 'val' if split == 'validation' else split
        index_path = self.resolve(component, f'endpoint_graphs/index_{cache}.pt')
        index = torch.load(index_path, map_location='cpu', weights_only=True)
        if graph_index >= len(index['shard_id']):
            raise IndexError('graph_index exceeds the stored split')
        shard_id = int(index['shard_id'][graph_index])
        local_idx = int(index['local_idx'][graph_index])
        relative = f'endpoint_graphs/{cache}-{shard_id:05d}.pt'
        if not (component / relative).is_file():
            manifest = json.loads((component / 'manifest.json').read_text())
            graphs = manifest.get('components', {}).get('guided', {}).get('endpoint_graphs', {})
            shards = graphs.get(cache, {}).get('artifacts', {}).get('shards', [])
            matching = [s['path'] for i, s in enumerate(shards) if s.get('shard_id', i) == shard_id]
            if len(matching) != 1:
                raise ValueError(f'No unique packaged location for shard {shard_id}')
            relative = matching[0]
        with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            data, slices = torch.load(self.resolve(component, relative), map_location='cpu', weights_only=True)
        if slices is None:
            if local_idx != 0:
                raise IndexError('A single-graph shard requires local_idx=0')
            return data.clone()
        return separate(cls=data.__class__, batch=data, idx=local_idx,
                        slice_dict=slices, decrement=False).clone()

    def validate(self):
        """Check file presence, sizes, portable graph references and sample parsing."""
        errors=[]
        for entry in self.catalog['files']:
            try:
                path=self.resolve(self.root,entry['path'])
                if path.stat().st_size != entry['bytes']:
                    errors.append('size mismatch: '+entry['path'])
            except (ValueError,FileNotFoundError) as exc:
                errors.append(str(exc))
        samples=[]
        graph_pointers=0
        seen=set()
        for name,entry in self.catalog['datasets'].items():
            if entry['path'] in seen:continue
            seen.add(entry['path'])
            component=self.dataset(name)
            mp=component/'manifest.json'
            if mp.is_file():
                manifest=json.loads(mp.read_text())
                graphs=manifest.get('components',{}).get('guided',{}).get('endpoint_graphs',{})
                for split,spec in graphs.items():
                    for shard in spec.get('artifacts',{}).get('shards',[]):
                        try:self.resolve(component,shard['path']);graph_pointers+=1
                        except (TypeError,ValueError,FileNotFoundError) as exc:errors.append(name+': '+str(exc))
            role=entry['role']
            kind='groups' if role in ('canonical_guidance','structure_normalized') else 'routes'
            if role == 'frozen_evaluation':kind='evaluation'
            if role == 'candidate_source':kind='cases'
            if role in ('vocabulary','reaction_centers'):continue
            for split in ('train','validation','test'):
                try:
                    row=next(self.rows(name,split,kind))
                    samples.append({'dataset':name,'split':split,'kind':kind,'keys':sorted(row)[:12]})
                except KeyError:
                    # Sealed views intentionally expose only their registered split.
                    continue
                except (StopIteration,ValueError,FileNotFoundError) as exc:
                    errors.append(name+'/'+split+': '+repr(exc))
        return {'status':'passed' if not errors else 'failed','files':len(self.catalog['files']),
                'sampled_views':len(samples),'graph_shard_pointers':graph_pointers,
                'checks':['file presence and sizes','graph shard path resolution','first-record JSON/CSV parsing'],
                'not_run':['model execution','full training','inference','statistical recomputation','content hashes'],
                'samples':samples,'errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--split',default='train',choices=['train','validation','val','test'])
    parser.add_argument('--kind',default='groups',choices=['groups','routes','raw','cases','evaluation','metrics','shuffle','pools'])
    parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    package=DataPackage(args.root)
    if args.list:
        for name,entry in package.catalog['datasets'].items():print(name,entry['role'],entry['path'])
    elif args.validate:
        result=package.validate();print(json.dumps(result,indent=2));raise SystemExit(bool(result['errors']))
    elif args.dataset:
        if args.limit < 1:parser.error('--limit must be positive')
        for row in itertools.islice(package.rows(args.dataset,args.split,args.kind),args.limit):
            print(json.dumps(row,ensure_ascii=False))
    else:parser.error('choose --list, --validate or --dataset')


if __name__=='__main__':main()
