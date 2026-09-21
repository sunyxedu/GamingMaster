import unittest
from scoring import Score, parse_inventory, LOGS, WEIGHTS
class ScoringTests(unittest.TestCase):
    def test_caps_and_terminal(self):
        s=Score();items={name:1 for name in WEIGHTS if name!='cooked_food'}
        items.update(oak_log=100,cobblestone=100,coal=2,charcoal=100,raw_iron=100,cooked_cod=1)
        s.update(items,4)
        self.assertEqual(s.report(300,300,20,'time_limit')['score'],100)
        self.assertEqual(s.report(300,300,20,None)['score'],90)
        self.assertEqual(s.report(150,300,0,'death')['score'],65)
        s.update({},5)
        self.assertEqual(s.report(300,300,20,'time_limit')['score'],100)
        self.assertEqual(s.milestones['cooked_food'],4)
    def test_idle(self):
        self.assertEqual(Score().report(300,300,20,'time_limit')['score'],20)
        self.assertEqual(Score().report(300,300,10,'time_limit')['score'],15)
        self.assertEqual(Score().report(100,300,0,'death')['score'],0)
    def test_combined_max_not_lifetime(self):
        s=Score();s.update({'oak_log':5,'coal':3},1);s.update({},2);s.update({'oak_log':5,'charcoal':3,'stripped_oak_log':64,'crimson_stem':64},3)
        self.assertEqual(s.maxima,{'logs':5,'cobblestone':0,'coal_charcoal':3,'raw_iron':0})
        self.assertEqual(s.report(0,300,20,None)['score'],8)
    def test_atomic_nbt_nested_and_excluded_slots(self):
        data='[{Slot: 0b, id: "minecraft:oak_log", count: 3}, {components: {"minecraft:container": [{item: {id: "minecraft:iron_ingot", count: 64}}]}, Slot: 1b, count: 1, id: "minecraft:shulker_box"}, {Slot: -106b, id: "minecraft:shield", count: 1}, {Slot: 100b, id: "minecraft:iron_boots", count: 1}, {Slot: 80b, id: "minecraft:diamond", count: 64}]'
        self.assertEqual(parse_inventory(data),{'oak_log':3,'shulker_box':1,'shield':1,'iron_boots':1})
        self.assertEqual(parse_inventory('[]'),{})
if __name__=='__main__':unittest.main()
