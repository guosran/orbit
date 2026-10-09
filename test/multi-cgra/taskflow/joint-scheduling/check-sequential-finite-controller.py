#!/usr/bin/env python3
"""Check receipts from actual production controllers; mutation tests use no costs.

Usage: check-sequential-finite-controller.py --search DIR [--domain JSON]
Checks objective debit uniqueness, fixed phase allowances, graph freezing,
finite-domain membership and immutable unaffected shapes. Does not implement
search or synthesize mapper/scheduler outcomes.
"""
import argparse,json
from pathlib import Path

def graph_actions(history):
 return history.get('fissionActions',[])+[a for a in history.get('actions',[]) if a['family'] not in ('shape','replica')]
def signature(actions):return json.dumps(actions,sort_keys=True)
def validate(summary,trace,domain=None):
 assert len(trace)==summary['production_scheduler_calls']<=summary['max_candidates']
 assert [x['cumulative_evaluations'] for x in trace]==list(range(1,len(trace)+1))
 assert len({x['key'] for x in trace})==len(trace),'duplicate objective debit'
 if summary['decision_flow']=='sequential':
  a=[x for x in trace if x['phase']=='A'];b=[x for x in trace if x['phase']=='B']
  assert len(a)+len(b)==len(trace)
  assert len(a)<=summary['phase_a_budget'] and len(b)<=summary['phase_b_budget']
  assert all(x['phase']=='A' for x in trace[:len(a)])
  assert all(all(s['rows']==s['cols']==1 for s in x['shapes']) for x in a),'phase A changed resources'
  anchor=next(x for x in a if x['candidate_id']==summary['phase_a_anchor'])
  graph=signature(graph_actions(anchor['action_history']))
  assert all(signature(graph_actions(x['action_history']))==graph for x in b),'frozen graph changed'
 if domain:
  legal={signature([a]) for a in domain['graph_actions']};legal.add(signature([]))
  for row in trace:
   assert signature(graph_actions(row['action_history'])) in legal,'graph outside finite domain'
   for shape in row['shapes']:
    pair=[shape['rows'],shape['cols']];name=shape['task']
    if name in domain['fixed_shapes']:assert pair==domain['fixed_shapes'][name],'unaffected task shape changed'
    else:
     assert any(name==root or name.startswith(root+'.') for root in domain['variable_tasks'])
     assert pair in domain['shapes'],'shape outside alphabet'

def main():
 p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);p.add_argument('--domain',type=Path);a=p.parse_args()
 summary=json.loads((a.search/'search-summary.json').read_text());trace=[json.loads(x) for x in (a.search/'budget-trace.jsonl').read_text().splitlines() if x.strip()]
 validate(summary,trace,json.loads(a.domain.read_text()) if a.domain else None)
 print('Production controller receipts verified:',len(trace),'unique objective evaluations')
if __name__=='__main__':main()
