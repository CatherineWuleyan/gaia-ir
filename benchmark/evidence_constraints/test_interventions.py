import copy, importlib.util, json, pathlib, unittest
P=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('interventions',P/'interventions.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Audit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cs=m.load()
 def test_complete_design(self):
  self.assertEqual(len(self.cs),24);self.assertEqual(len({c['question_id'] for c in self.cs}),24)
  for case in ('P1','P2','P3','P4'):self.assertEqual({c['variant'] for c in self.cs if c['case_id']==case},set(m.KINDS))
 def test_source_anchor_exact(self):
  for c in self.cs:
   for e in c['evidence']:
    if e['source']['path']=='synthetic':self.assertTrue(e['text'].startswith('【人工构造'))
    else:
     data=json.loads((m.REPO/e['source']['path']).read_text())
     def strings(x):
      if isinstance(x,str):yield x
      elif isinstance(x,dict):
       for v in x.values():yield from strings(v)
      elif isinstance(x,list):
       for v in x:yield from strings(v)
     self.assertTrue(e['source']['excerpt'] in set(strings(data)),e['id'])
 def test_no_gold_or_variant_to_model(self):
  for c in self.cs:
   shadow=copy.deepcopy(c);shadow['gold']={'status':'SECRET_GOLD'};shadow['change']='SECRET_CHANGE';shadow['variant']='SECRET_VARIANT';shadow['case_id']='SECRET_CASE'
   for cond in m.CONDS:self.assertEqual(m.public(c,cond),m.public(shadow,cond))
 def test_all_formats_same_evidence_and_relations(self):
  for c in self.cs:
   for cond in m.CONDS:
    p=m.public(c,cond)
    for e in c['evidence']:
     text=e['text'] if cond=='raw_rag' else json.dumps(e['text'],ensure_ascii=False)[1:-1]
     self.assertTrue(text in p,e['id']);self.assertTrue(e['origin'] in p,e['id'])
    for e in c['relations']:
     for value in e.values():self.assertIn(value,p)
 def test_removed_evidence_does_not_leak(self):
  for c in self.cs:
   for e in c['change']['removed']:
    for cond in m.CONDS:self.assertNotIn(e['text'],m.public(c,cond))
   for cond in m.CONDS:
    p=m.public(c,cond)
    self.assertNotIn('M2',p);self.assertNotIn('R2',p)
 def test_duplicates_not_independent(self):
  for c in self.cs:
   if c['variant']=='duplicate':
    b=next(b for b in self.cs if b['case_id']==c['case_id'] and b['variant']=='baseline')
    self.assertEqual(c['gold']['independent_origins'],b['gold']['independent_origins']);self.assertEqual(len(c['evidence']),2*len(b['evidence']))
 def test_counter_not_paper_fact(self):
  for c in self.cs:
   if c['variant']=='counterexample':
    for e in c['change']['added']:self.assertEqual(e['kind'],'synthetic_observation');self.assertIn('不是论文事实',e['text'])
 def test_expected_transitions(self):
  for c in self.cs:
   expected='unknown' if c['variant'] in ('delete','condition') or (c['case_id']=='P2' and c['variant'] in ('baseline','irrelevant','duplicate')) else ('conflict' if c['variant']=='counterexample' else 'supported')
   self.assertEqual(c['gold']['status'],expected)
  self.assertAlmostEqual(31/5.5,5.63636363636);self.assertLess(3*5.5,31)
 def test_malformed_failures_not_unknown_success(self):
  c=next(c for c in self.cs if c['gold']['status']=='unknown')
  self.assertFalse(m.score(c,None)['correct']);self.assertEqual(m.score(c,None)['prediction'],'error')
  self.assertFalse(m.score(c,{'status':'unknown'})['joint'])
 def test_wrong_count_and_invented_citation_fail(self):
  c=self.cs[0];ids=[e['id'] for e in c['evidence']]
  r=dict(status=c['gold']['status'],answer=c['gold']['answer'],premises=ids,provenance=ids,evidence_path=[],conditions=[{'key':'independent_origins','value':c['gold']['independent_origins']}])
  self.assertTrue(m.score(c,r)['joint']);r['premises'].append('invented');self.assertFalse(m.score(c,r)['joint']);r['premises'].pop();r['conditions'][0]['value']+=1;self.assertFalse(m.score(c,r)['joint'])
if __name__=='__main__':unittest.main()
