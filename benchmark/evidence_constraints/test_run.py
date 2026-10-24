import collections, importlib.util, json, pathlib, unittest

HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot_run',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class EvaluationContracts(unittest.TestCase):
    def test_paper_disjoint_and_balanced(self):
        dev=r.rows(HERE/'dev.jsonl');test=r.rows(HERE/'test.jsonl')
        self.assertEqual((len(dev),len(test)),(6,12))
        self.assertFalse({q['source'] for q in dev}&{q['source'] for q in test})
        for items,n in ((dev,2),(test,4)):
            self.assertEqual(collections.Counter(q['gold']['status'] for q in items),dict.fromkeys(r.LABELS,n))
        texts={}
        for q in dev+test:
            text=(r.REPO/q['source']).read_text()
            h=r.sha(text)
            self.assertTrue(h not in texts or texts[h]==q['source'])
            texts[h]=q['source']

    def test_gold_is_excluded_and_flat_graph_content_matched(self):
        for q in r.rows(HERE/'dev.jsonl')+r.rows(HERE/'test.jsonl'):
            c=r.corpus(q['source'])
            public={'question_id':q['question_id'],'question':q['question']}
            _,b=r.public_input(public,c,'flat_claims');_,g=r.public_input(public,c,'graph_context')
            self.assertEqual(b['structured_content_sha256'],g['structured_content_sha256'])
            self.assertEqual(b['source_ids'],g['source_ids'])
            for cond in r.CONDITIONS:
                p,_=r.public_input(public,c,cond)
                self.assertNotIn('"gold"',p)
                self.assertNotIn('required_premises',p)
                self.assertNotIn('gold_status',p)

    def test_unknown_is_not_transport_failure(self):
        q={'gold':{'status':'unknown','answer':None,'required_premises':[],'conditions':[]}}
        m={'source_ids':['p1']}
        self.assertFalse(r.score(q,None,m)['status_correct'])
        self.assertTrue(r.score(q,{'status':'unknown','answer':None},m)['joint_success'])
        self.assertFalse(r.score(q,{'status':'unknown','answer':None,'premises':['fake']},m)['joint_success'])

    def test_wrong_or_missing_scope_and_fabricated_source_fail(self):
        q={'gold':{'status':'supported','answer':'yes','required_premises':['p1'],'conditions':[{'key':'dataset','value':'CIFAR-10'}]}}
        m={'source_ids':['p1']}
        response={'status':'supported','answer':'yes','premises':['p1'],'conditions':[{'key':'dataset','value':'CIFAR-10'}]}
        self.assertTrue(r.score(q,response,m)['joint_success'])
        response['conditions']=[];self.assertFalse(r.score(q,response,m)['joint_success'])
        response['provenance']=['fake'];self.assertFalse(r.score(q,response,m)['source_ids_valid'])

    def test_gold_ids_point_to_existing_source_spans(self):
        for q in r.rows(HERE/'dev.jsonl')+r.rows(HERE/'test.jsonl'):
            c=r.corpus(q['source']);ids={p['id'] for p in c['spans']}
            self.assertTrue(set(q['gold']['required_premises']).issubset(ids))

if __name__=='__main__':unittest.main()
