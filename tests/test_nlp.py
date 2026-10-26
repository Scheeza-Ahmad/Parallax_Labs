import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.nlp import discover_topics, extract_entities, metadata_for, make_filter, evaluate_entities
from src.vector_store import client_at, open_collection, search


def test_topics_deterministic_and_short_unknown():
    texts = ['ocean water fish waves coral '*8, 'ocean water fish beach '*8,
             'computer program software code '*8, 'computer program software binary '*8,
             'tiny', 'zzq '*25]
    a = discover_topics(texts, 2, 2)
    b = discover_topics(texts, 2, 2)
    assert a[2] == b[2]
    assert a[2][-2]['topic_status'] == 'short'
    assert a[2][-1]['topic_status'] == 'out_of_vocabulary'
    assert a[2][-1]['topic_id'] == -1
    assert all(0 <= x['topic_weight'] <= 1 for x in a[2])


def test_empty_vocabulary_is_actionable():
    with pytest.raises(ValueError, match='vocabulary'):
        discover_topics(['the and', 'and the'], 2)


def test_exact_span_metrics_and_invalid_gold():
    nlp = lambda text: SimpleNamespace(ents=[SimpleNamespace(start_char=0,end_char=5,label_='PERSON',text='Alice')])
    examples = [{'doc_id':'1','text':'Alice in Rome','entities':[
        {'start':0,'end':5,'label':'PERSON','text':'Alice'},
        {'start':9,'end':13,'label':'GPE','text':'Rome'}]}]
    scores = evaluate_entities(nlp, examples)
    assert scores['precision'] == 1 and scores['recall'] == .5
    assert scores['f1'] == pytest.approx(2/3)
    assert scores['exact_example_accuracy'] == 0
    examples[0]['entities'][0]['text'] = 'Bob'
    with pytest.raises(ValueError, match='offsets'):
        evaluate_entities(nlp, examples)


def test_long_text_offsets_and_no_truncation():
    seen=[]
    def nlp(text):
        seen.append(len(text))
        return SimpleNamespace(ents=[SimpleNamespace(start_char=0,end_char=1,label_='ORG',text=text[0])])
    entities = extract_entities(nlp, 'x'*50001)
    assert seen == [50000,1] and entities[1]['start'] == 50000
    assert metadata_for({'topic_id':0,'topic_weight':.7,'topic_status':'assigned','entities':entities})['has_org'] is True


@pytest.mark.parametrize('kwargs', [{'topic':-2},{'entity':'INVALID'},{'min_weight':1.1},{'min_weight':float('nan')}])
def test_invalid_filters(kwargs):
    with pytest.raises(ValueError): make_filter(**kwargs)


def test_real_chroma_filter_and_zero_matches(tmp_path):
    class Encoder:
        identity={'model':'fixture','dimension':2}
        def encode(self, texts): return np.array([[1.,0.]])
    c,_=open_collection(client_at(tmp_path/'db'),'nlp_test',{'encoder':Encoder.identity},True)
    c.add(ids=['a','b'],documents=['one','two'],embeddings=[[1.,0.],[0.,1.]],
          metadatas=[{'topic_id':1,'has_person':True,'topic_weight':.8},
                     {'topic_id':2,'has_person':False,'topic_weight':.9}])
    c.modify(metadata=dict(c.metadata,status='ready'))
    r=search(c,Encoder(),'q',5,where=make_filter(1,'PERSON',.5))
    assert [h['chunk_id'] for h in r['hits']] == ['a']
    assert search(c,Encoder(),'q',5,where=make_filter(999))['returned'] == 0


def test_enrichment_preserves_source_resumes_and_rejects_changes(tmp_path):
    from scripts.nlp import enrich
    client=client_at(tmp_path/'db')
    source,_=open_collection(client,'source_test',{'corpus_sha256':'abc','documents':1,'encoder':{'dimension':2}},True)
    source.add(ids=['one'],documents=['Alice lives here'],metadatas=[{'doc_id':'1','confidence':.9999978173055841}],embeddings=[[1.,0.]])
    source.modify(metadata=dict(source.metadata,status='ready'))
    out=tmp_path/'nlp'; out.mkdir()
    artifact={'corpus_sha256':'abc','records':[{'doc_id':'1','topic_id':0,'topic_weight':.7,
               'topic_status':'assigned','entities':[]}]}
    (out/'metadata.json').write_text(json.dumps(artifact))
    args=SimpleNamespace(output=out,db=tmp_path/'db',source='source_test',collection='enriched_test',report=tmp_path/'report.json')
    enrich(args); enrich(args)
    assert json.loads(args.report.read_text())['skipped']==1
    assert 'topic_id' not in source.get(include=['metadatas'])['metadatas'][0]
    # Persistence must work in a new interpreter, not just the writer's cache.
    import subprocess
    import sys
    code = ('from src.vector_store import client_at; import sys; '
            "c=client_at(sys.argv[1]).get_collection('enriched_test',embedding_function=None); "
            "assert c.count()==1; assert c.query(query_embeddings=[[1.,0.]],n_results=1,where={'topic_id':0})['ids']==[['one']]")
    subprocess.run([sys.executable, '-c', code, str(args.db)], check=True,
                   capture_output=True, text=True, timeout=120)
    artifact['records'][0]['topic_id']=1
    (out/'metadata.json').write_text(json.dumps(artifact))
    with pytest.raises(ValueError,match='Incompatible'): enrich(args)
    artifact['corpus_sha256']='wrong'
    (out/'metadata.json').write_text(json.dumps(artifact))
    with pytest.raises(ValueError,match='corpus'): enrich(args)
