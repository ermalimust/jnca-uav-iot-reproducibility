"""Symmetric, outcome-blind serialization normalization approved before evaluation.

Accept ONLY a one-to-one list representation of exactly the six archetype keys.
Never alter candidate strings/order, complete missing keys, or repair duplicates.
Original literal outputs, strict parsing and tool observations remain untouched.
"""
import copy
from react_public import ARCH
def normalize(policy):
    out=copy.deepcopy(policy)
    if not isinstance(out,dict):return out,'unchanged_non_object'
    groups=out.get('archetype_actions')
    if isinstance(groups,dict):return out,'unchanged_object'
    if not isinstance(groups,list):return out,'unchanged_non_equivalent'
    if len(groups)!=6:return out,'unchanged_non_equivalent'
    if not all(isinstance(x,dict) and set(x)=={'archetype','actions'} and isinstance(x['archetype'],str) and isinstance(x['actions'],list) for x in groups):return out,'unchanged_non_equivalent'
    names=[x['archetype'] for x in groups]
    if len(set(names))!=6 or set(names)!=set(ARCH):return out,'unchanged_non_equivalent'
    out['archetype_actions']={x['archetype']:x['actions'] for x in groups}
    return out,'equivalent_list_to_object'

def selfcheck():
    groups=[{'archetype':a,'actions':['Unknown','Observe','Observe']} for a in ARCH]
    original={'task_id':'TTEST','archetype_actions':groups};out,status=normalize(original)
    assert status=='equivalent_list_to_object'
    assert all(out['archetype_actions'][a]==['Unknown','Observe','Observe'] for a in ARCH)
    assert isinstance(original['archetype_actions'],list)
    for variant in [groups[:-1],groups+[groups[0]],groups[:-1]+[groups[0]],[{**x,'extra':True} for x in groups]]:
        p={'archetype_actions':variant};got,state=normalize(p);assert got==p and state=='unchanged_non_equivalent'
    assert normalize(out)==(out,'unchanged_object')
    assert normalize(None)==(None,'unchanged_non_object')
    return {'one_to_one_only':True,'exact_six_unique_keys':True,'unknown_strings_and_duplicates_preserved':True,'missing_duplicate_extra_fields_rejected':True,'candidate_order_preserved':True,'input_unchanged':True,'all_passed':True}
