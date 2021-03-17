# Autogen with
# import json
# import pprint
# def findall(string, sub):
#     idxes = []
#     start = 0
#     while True:
#         idx = string.find(sub, start)
#         if idx == -1:
#             break
#         idxes.append(idx)
#         start = idx + len(sub)
#     return idxes
# def convert_quotas(string):
#     idxes = findall(string, "'")
#     for left, right in zip(idxes[0::2], idxes[1::2]):
#         string = string.replace(string[left], "\"", 1)
#         string = string.replace(string[right], "\"", 1)
#     return string
# a = json.load(open("objects365_categories.json"))
# string = ""
# string = pprint.pformat(
#     a,
#     indent=4,
#     width=100,
#     compact=True).replace("{", "{\n" + " " * 5).replace("}", "\n" + " " * 4 + "}")
# string = string[:1] + "\n " + string[1:]
# string = "LVIS_CATEGORIES = " + string[:-1] + "\n]"
# flag = False
# with open("lvis_categories.py", mode="w+") as f:
#     for line in string.split("\n"):
#         if "[   " in line:
#             line = line.replace("[   ", "[")
#             flag = True
#         elif flag:
#             line = line[3:]  # align with last line
#             flag = False
#         print(convert_quotas(line), file=f)


LVIS_CATEGORIES = [
    {
        "def": "nut from an oak tree",
        "frequency": "r",
        "id": 1,
        "name": "acorn",
        "synonyms": ["acorn"],
        "synset": "acorn.n.01"
    },
    {
        "def": "a dispenser that holds a substance under pressure",
        "frequency": "c",
        "id": 2,
        "name": "aerosol_can",
        "synonyms": ["aerosol_can", "spray_can"],
        "synset": "aerosol.n.02"
    },
    {
        "def": "a machine that keeps air cool and dry",
        "frequency": "f",
        "id": 3,
        "name": "air_conditioner",
        "synonyms": ["air_conditioner"],
        "synset": "air_conditioner.n.01"
    },
    {
        "def": "an aircraft that has a fixed wing and is powered by propellers or jets",
        "frequency": "f",
        "id": 4,
        "name": "airplane",
        "synonyms": ["airplane", "aeroplane"],
        "synset": "airplane.n.01"
    },
    {
        "def": "a clock that wakes a sleeper at some preset time",
        "frequency": "c",
        "id": 5,
        "name": "alarm_clock",
        "synonyms": ["alarm_clock"],
        "synset": "alarm_clock.n.01"
    },
    {
        "def": "a liquor or brew containing alcohol as the active agent",
        "frequency": "c",
        "id": 6,
        "name": "alcohol",
        "synonyms": ["alcohol", "alcoholic_beverage"],
        "synset": "alcohol.n.01"
    },
    {
        "def": "amphibious reptiles related to crocodiles but with shorter broader snouts",
        "frequency": "r",
        "id": 7,
        "name": "alligator",
        "synonyms": ["alligator", "gator"],
        "synset": "alligator.n.02"
    },
    {
        "def": "oval-shaped edible seed of the almond tree",
        "frequency": "c",
        "id": 8,
        "name": "almond",
        "synonyms": ["almond"],
        "synset": "almond.n.02"
    },
    {
        "def": "a vehicle that takes people to and from hospitals",
        "frequency": "c",
        "id": 9,
        "name": "ambulance",
        "synonyms": ["ambulance"],
        "synset": "ambulance.n.01"
    },
    {
        "def": "electronic equipment that increases strength of signals",
        "frequency": "r",
        "id": 10,
        "name": "amplifier",
        "synonyms": ["amplifier"],
        "synset": "amplifier.n.01"
    },
    {
        "def": "an ornament worn around the ankle",
        "frequency": "c",
        "id": 11,
        "name": "anklet",
        "synonyms": ["anklet", "ankle_bracelet"],
        "synset": "anklet.n.03"
    },
    {
        "def": "an electrical device that sends or receives radio or television signals",
        "frequency": "f",
        "id": 12,
        "name": "antenna",
        "synonyms": ["antenna", "aerial", "transmitting_aerial"],
        "synset": "antenna.n.01"
    },
    {
        "def": "fruit with red or yellow or green skin and sweet to tart crisp whitish flesh",
        "frequency": "f",
        "id": 13,
        "name": "apple",
        "synonyms": ["apple"],
        "synset": "apple.n.01"
    },
    {
        "def": "the juice of apples",
        "frequency": "r",
        "id": 14,
        "name": "apple_juice",
        "synonyms": ["apple_juice"],
        "synset": "apple_juice.n.01"
    },
    {
        "def": "puree of stewed apples usually sweetened and spiced",
        "frequency": "r",
        "id": 15,
        "name": "applesauce",
        "synonyms": ["applesauce"],
        "synset": "applesauce.n.01"
    },
    {
        "def": "downy yellow to rosy-colored fruit resembling a small peach",
        "frequency": "r",
        "id": 16,
        "name": "apricot",
        "synonyms": ["apricot"],
        "synset": "apricot.n.02"
    },
    {
        "def": "a garment of cloth that is tied about the waist and worn to protect clothing",
        "frequency": "f",
        "id": 17,
        "name": "apron",
        "synonyms": ["apron"],
        "synset": "apron.n.01"
    },
    {
        "def": "a tank/pool/bowl filled with water for keeping live fish and underwater animals",
        "frequency": "c",
        "id": 18,
        "name": "aquarium",
        "synonyms": ["aquarium", "fish_tank"],
        "synset": "aquarium.n.01"
    },
    {
        "def": "a band worn around the upper arm",
        "frequency": "c",
        "id": 19,
        "name": "armband",
        "synonyms": ["armband"],
        "synset": "armband.n.02"
    },
    {
        "def": "chair with a support on each side for arms",
        "frequency": "f",
        "id": 20,
        "name": "armchair",
        "synonyms": ["armchair"],
        "synset": "armchair.n.01"
    },
    {
        "def": "a large wardrobe or cabinet",
        "frequency": "r",
        "id": 21,
        "name": "armoire",
        "synonyms": ["armoire"],
        "synset": "armoire.n.01"
    },
    {
        "def": "protective covering made of metal and used in combat",
        "frequency": "r",
        "id": 22,
        "name": "armor",
        "synonyms": ["armor", "armour"],
        "synset": "armor.n.01"
    },
    {
        "def": "a thistlelike flower head with edible fleshy leaves and heart",
        "frequency": "c",
        "id": 23,
        "name": "artichoke",
        "synonyms": ["artichoke"],
        "synset": "artichoke.n.02"
    },
    {
        "def": "a bin that holds rubbish until it is collected",
        "frequency": "f",
        "id": 24,
        "name": "trash_can",
        "synonyms": ["trash_can", "garbage_can", "wastebin", "dustbin", "trash_barrel",
                     "trash_bin"],
        "synset": "ashcan.n.01"
    },
    {
        "def": "a receptacle for the ash from smokers' cigars or cigarettes",
        "frequency": "c",
        "id": 25,
        "name": "ashtray",
        "synonyms": ["ashtray"],
        "synset": "ashtray.n.01"
    },
    {
        "def": "edible young shoots of the asparagus plant",
        "frequency": "c",
        "id": 26,
        "name": "asparagus",
        "synonyms": ["asparagus"],
        "synset": "asparagus.n.02"
    },
    {
        "def": "a dispenser that turns a liquid (such as perfume) into a fine mist",
        "frequency": "c",
        "id": 27,
        "name": "atomizer",
        "synonyms": ["atomizer", "atomiser", "spray", "sprayer", "nebulizer", "nebuliser"],
        "synset": "atomizer.n.01"
    },
    {
        "def": "a pear-shaped fruit with green or blackish skin and rich yellowish pulp enclosing "
               "a single large seed",
        "frequency": "c",
        "id": 28,
        "name": "avocado",
        "synonyms": ["avocado"],
        "synset": "avocado.n.01"
    },
    {
        "def": "a tangible symbol signifying approval or distinction",
        "frequency": "c",
        "id": 29,
        "name": "award",
        "synonyms": ["award", "accolade"],
        "synset": "award.n.02"
    },
    {
        "def": "a canopy made of canvas to shelter people or things from rain or sun",
        "frequency": "f",
        "id": 30,
        "name": "awning",
        "synonyms": ["awning"],
        "synset": "awning.n.01"
    },
    {
        "def": "an edge tool with a heavy bladed head mounted across a handle",
        "frequency": "r",
        "id": 31,
        "name": "ax",
        "synonyms": ["ax", "axe"],
        "synset": "ax.n.01"
    },
    {
        "def": "a small vehicle with four wheels in which a baby or child is pushed around",
        "frequency": "f",
        "id": 32,
        "name": "baby_buggy",
        "synonyms": ["baby_buggy", "baby_carriage", "perambulator", "pram", "stroller"],
        "synset": "baby_buggy.n.01"
    },
    {
        "def": "a raised vertical board with basket attached; used to play basketball",
        "frequency": "c",
        "id": 33,
        "name": "basketball_backboard",
        "synonyms": ["basketball_backboard"],
        "synset": "backboard.n.01"
    },
    {
        "def": "a bag carried by a strap on your back or shoulder",
        "frequency": "f",
        "id": 34,
        "name": "backpack",
        "synonyms": ["backpack", "knapsack", "packsack", "rucksack", "haversack"],
        "synset": "backpack.n.01"
    },
    {
        "def": "a container used for carrying money and small personal items or accessories",
        "frequency": "f",
        "id": 35,
        "name": "handbag",
        "synonyms": ["handbag", "purse", "pocketbook"],
        "synset": "bag.n.04"
    },
    {
        "def": "cases used to carry belongings when traveling",
        "frequency": "f",
        "id": 36,
        "name": "suitcase",
        "synonyms": ["suitcase", "baggage", "luggage"],
        "synset": "bag.n.06"
    },
    {
        "def": "glazed yeast-raised doughnut-shaped roll with hard crust",
        "frequency": "c",
        "id": 37,
        "name": "bagel",
        "synonyms": ["bagel", "beigel"],
        "synset": "bagel.n.01"
    },
    {
        "def": "a tubular wind instrument; the player blows air into a bag and squeezes it out",
        "frequency": "r",
        "id": 38,
        "name": "bagpipe",
        "synonyms": ["bagpipe"],
        "synset": "bagpipe.n.01"
    },
    {
        "def": "narrow French stick loaf",
        "frequency": "r",
        "id": 39,
        "name": "baguet",
        "synonyms": ["baguet", "baguette"],
        "synset": "baguet.n.01"
    },
    {
        "def": "something used to lure fish or other animals into danger so they can be trapped or "
               "killed",
        "frequency": "r",
        "id": 40,
        "name": "bait",
        "synonyms": ["bait", "lure"],
        "synset": "bait.n.02"
    },
    {
        "def": "a spherical object used as a plaything",
        "frequency": "f",
        "id": 41,
        "name": "ball",
        "synonyms": ["ball"],
        "synset": "ball.n.06"
    },
    {
        "def": "very short skirt worn by ballerinas",
        "frequency": "r",
        "id": 42,
        "name": "ballet_skirt",
        "synonyms": ["ballet_skirt", "tutu"],
        "synset": "ballet_skirt.n.01"
    },
    {
        "def": "large tough nonrigid bag filled with gas or heated air",
        "frequency": "f",
        "id": 43,
        "name": "balloon",
        "synonyms": ["balloon"],
        "synset": "balloon.n.01"
    },
    {
        "def": "woody tropical grass having hollow woody stems",
        "frequency": "c",
        "id": 44,
        "name": "bamboo",
        "synonyms": ["bamboo"],
        "synset": "bamboo.n.02"
    },
    {
        "def": "elongated crescent-shaped yellow fruit with soft sweet flesh",
        "frequency": "f",
        "id": 45,
        "name": "banana",
        "synonyms": ["banana"],
        "synset": "banana.n.02"
    },
    {
        "def": "trade name for an adhesive bandage to cover small cuts or blisters",
        "frequency": "r",
        "id": 46,
        "name": "Band_Aid",
        "synonyms": ["Band_Aid"],
        "synset": "band_aid.n.01"
    },
    {
        "def": "a piece of soft material that covers and protects an injured part of the body",
        "frequency": "c",
        "id": 47,
        "name": "bandage",
        "synonyms": ["bandage"],
        "synset": "bandage.n.01"
    },
    {
        "def": "large and brightly colored handkerchief; often used as a neckerchief",
        "frequency": "c",
        "id": 48,
        "name": "bandanna",
        "synonyms": ["bandanna", "bandana"],
        "synset": "bandanna.n.01"
    },
    {
        "def": "a stringed instrument of the guitar family with a long neck and circular body",
        "frequency": "r",
        "id": 49,
        "name": "banjo",
        "synonyms": ["banjo"],
        "synset": "banjo.n.01"
    },
    {
        "def": "long strip of cloth or paper used for decoration or advertising",
        "frequency": "f",
        "id": 50,
        "name": "banner",
        "synonyms": ["banner", "streamer"],
        "synset": "banner.n.01"
    },
    {
        "def": "a bar to which heavy discs are attached at each end; used in weightlifting",
        "frequency": "r",
        "id": 51,
        "name": "barbell",
        "synonyms": ["barbell"],
        "synset": "barbell.n.01"
    },
    {
        "def": "a flatbottom boat for carrying heavy loads (especially on canals)",
        "frequency": "r",
        "id": 52,
        "name": "barge",
        "synonyms": ["barge"],
        "synset": "barge.n.01"
    },
    {
        "def": "a cylindrical container that holds liquids",
        "frequency": "f",
        "id": 53,
        "name": "barrel",
        "synonyms": ["barrel", "cask"],
        "synset": "barrel.n.02"
    },
    {
        "def": "a pin for holding women's hair in place",
        "frequency": "c",
        "id": 54,
        "name": "barrette",
        "synonyms": ["barrette"],
        "synset": "barrette.n.01"
    },
    {
        "def": "a cart for carrying small loads; has handles and one or more wheels",
        "frequency": "c",
        "id": 55,
        "name": "barrow",
        "synonyms": ["barrow", "garden_cart", "lawn_cart", "wheelbarrow"],
        "synset": "barrow.n.03"
    },
    {
        "def": "a place that the runner must touch before scoring",
        "frequency": "f",
        "id": 56,
        "name": "baseball_base",
        "synonyms": ["baseball_base"],
        "synset": "base.n.03"
    },
    {
        "def": "a ball used in playing baseball",
        "frequency": "f",
        "id": 57,
        "name": "baseball",
        "synonyms": ["baseball"],
        "synset": "baseball.n.02"
    },
    {
        "def": "an implement used in baseball by the batter",
        "frequency": "f",
        "id": 58,
        "name": "baseball_bat",
        "synonyms": ["baseball_bat"],
        "synset": "baseball_bat.n.01"
    },
    {
        "def": "a cap with a bill",
        "frequency": "f",
        "id": 59,
        "name": "baseball_cap",
        "synonyms": ["baseball_cap", "jockey_cap", "golf_cap"],
        "synset": "baseball_cap.n.01"
    },
    {
        "def": "the handwear used by fielders in playing baseball",
        "frequency": "f",
        "id": 60,
        "name": "baseball_glove",
        "synonyms": ["baseball_glove", "baseball_mitt"],
        "synset": "baseball_glove.n.01"
    },
    {
        "def": "a container that is usually woven and has handles",
        "frequency": "f",
        "id": 61,
        "name": "basket",
        "synonyms": ["basket", "handbasket"],
        "synset": "basket.n.01"
    },
    {
        "def": "metal hoop supporting a net through which players try to throw the basketball",
        "frequency": "c",
        "id": 62,
        "name": "basketball_hoop",
        "synonyms": ["basketball_hoop"],
        "synset": "basket.n.03"
    },
    {
        "def": "an inflated ball used in playing basketball",
        "frequency": "c",
        "id": 63,
        "name": "basketball",
        "synonyms": ["basketball"],
        "synset": "basketball.n.02"
    },
    {
        "def": "the lowest brass wind instrument",
        "frequency": "r",
        "id": 64,
        "name": "bass_horn",
        "synonyms": ["bass_horn", "sousaphone", "tuba"],
        "synset": "bass_horn.n.01"
    },
    {
        "def": "nocturnal mouselike mammal with forelimbs modified to form membranous wings",
        "frequency": "r",
        "id": 65,
        "name": "bat_(animal)",
        "synonyms": ["bat_(animal)"],
        "synset": "bat.n.01"
    },
    {
        "def": "a heavy towel or mat to stand on while drying yourself after a bath",
        "frequency": "f",
        "id": 66,
        "name": "bath_mat",
        "synonyms": ["bath_mat"],
        "synset": "bath_mat.n.01"
    },
    {
        "def": "a large towel; to dry yourself after a bath",
        "frequency": "f",
        "id": 67,
        "name": "bath_towel",
        "synonyms": ["bath_towel"],
        "synset": "bath_towel.n.01"
    },
    {
        "def": "a loose-fitting robe of towelling; worn after a bath or swim",
        "frequency": "c",
        "id": 68,
        "name": "bathrobe",
        "synonyms": ["bathrobe"],
        "synset": "bathrobe.n.01"
    },
    {
        "def": "a large open container that you fill with water and use to wash the body",
        "frequency": "f",
        "id": 69,
        "name": "bathtub",
        "synonyms": ["bathtub", "bathing_tub"],
        "synset": "bathtub.n.01"
    },
    {
        "def": "a liquid or semiliquid mixture, as of flour, eggs, and milk, used in cooking",
        "frequency": "r",
        "id": 70,
        "name": "batter_(food)",
        "synonyms": ["batter_(food)"],
        "synset": "batter.n.02"
    },
    {
        "def": "a portable device that produces electricity",
        "frequency": "c",
        "id": 71,
        "name": "battery",
        "synonyms": ["battery"],
        "synset": "battery.n.02"
    },
    {
        "def": "large and light ball; for play at the seaside",
        "frequency": "r",
        "id": 72,
        "name": "beachball",
        "synonyms": ["beachball"],
        "synset": "beach_ball.n.01"
    },
    {
        "def": "a small ball with a hole through the middle used for ornamentation, jewellery, "
               "etc.",
        "frequency": "c",
        "id": 73,
        "name": "bead",
        "synonyms": ["bead"],
        "synset": "bead.n.01"
    },
    {
        "def": "a flatbottomed jar made of glass or plastic; used for chemistry",
        "frequency": "r",
        "id": 74,
        "name": "beaker",
        "synonyms": ["beaker"],
        "synset": "beaker.n.01"
    },
    {
        "def": "cheeselike food made of curdled soybean milk",
        "frequency": "c",
        "id": 75,
        "name": "bean_curd",
        "synonyms": ["bean_curd", "tofu"],
        "synset": "bean_curd.n.01"
    },
    {
        "def": "a bag filled with dried beans or similar items; used in games or to sit on",
        "frequency": "c",
        "id": 76,
        "name": "beanbag",
        "synonyms": ["beanbag"],
        "synset": "beanbag.n.01"
    },
    {
        "def": "a small skullcap; formerly worn by schoolboys and college freshmen",
        "frequency": "f",
        "id": 77,
        "name": "beanie",
        "synonyms": ["beanie", "beany"],
        "synset": "beanie.n.01"
    },
    {
        "def": "large carnivorous or omnivorous mammals with shaggy coats and claws",
        "frequency": "f",
        "id": 78,
        "name": "bear",
        "synonyms": ["bear"],
        "synset": "bear.n.01"
    },
    {
        "def": "a piece of furniture that provides a place to sleep",
        "frequency": "f",
        "id": 79,
        "name": "bed",
        "synonyms": ["bed"],
        "synset": "bed.n.01"
    },
    {
        "def": "decorative cover for a bed",
        "frequency": "c",
        "id": 80,
        "name": "bedspread",
        "synonyms": ["bedspread", "bedcover", "bed_covering", "counterpane", "spread"],
        "synset": "bedspread.n.01"
    },
    {
        "def": "cattle that are reared for their meat",
        "frequency": "f",
        "id": 81,
        "name": "cow",
        "synonyms": ["cow"],
        "synset": "beef.n.01"
    },
    {
        "def": "meat from an adult domestic bovine",
        "frequency": "c",
        "id": 82,
        "name": "beef_(food)",
        "synonyms": ["beef_(food)", "boeuf_(food)"],
        "synset": "beef.n.02"
    },
    {
        "def": "an device that beeps when the person carrying it is being paged",
        "frequency": "r",
        "id": 83,
        "name": "beeper",
        "synonyms": ["beeper", "pager"],
        "synset": "beeper.n.01"
    },
    {
        "def": "a bottle that holds beer",
        "frequency": "f",
        "id": 84,
        "name": "beer_bottle",
        "synonyms": ["beer_bottle"],
        "synset": "beer_bottle.n.01"
    },
    {
        "def": "a can that holds beer",
        "frequency": "c",
        "id": 85,
        "name": "beer_can",
        "synonyms": ["beer_can"],
        "synset": "beer_can.n.01"
    },
    {
        "def": "insect with hard wing covers",
        "frequency": "r",
        "id": 86,
        "name": "beetle",
        "synonyms": ["beetle"],
        "synset": "beetle.n.01"
    },
    {
        "def": "a hollow device made of metal that makes a ringing sound when struck",
        "frequency": "f",
        "id": 87,
        "name": "bell",
        "synonyms": ["bell"],
        "synset": "bell.n.01"
    },
    {
        "def": "large bell-shaped sweet pepper in green or red or yellow or orange or black "
               "varieties",
        "frequency": "f",
        "id": 88,
        "name": "bell_pepper",
        "synonyms": ["bell_pepper", "capsicum"],
        "synset": "bell_pepper.n.02"
    },
    {
        "def": "a band to tie or buckle around the body (usually at the waist)",
        "frequency": "f",
        "id": 89,
        "name": "belt",
        "synonyms": ["belt"],
        "synset": "belt.n.02"
    },
    {
        "def": "the buckle used to fasten a belt",
        "frequency": "f",
        "id": 90,
        "name": "belt_buckle",
        "synonyms": ["belt_buckle"],
        "synset": "belt_buckle.n.01"
    },
    {
        "def": "a long seat for more than one person",
        "frequency": "f",
        "id": 91,
        "name": "bench",
        "synonyms": ["bench"],
        "synset": "bench.n.01"
    },
    {
        "def": "a cap with no brim or bill; made of soft cloth",
        "frequency": "c",
        "id": 92,
        "name": "beret",
        "synonyms": ["beret"],
        "synset": "beret.n.01"
    },
    {
        "def": "a napkin tied under the chin of a child while eating",
        "frequency": "c",
        "id": 93,
        "name": "bib",
        "synonyms": ["bib"],
        "synset": "bib.n.02"
    },
    {
        "def": "the sacred writings of the Christian religions",
        "frequency": "r",
        "id": 94,
        "name": "Bible",
        "synonyms": ["Bible"],
        "synset": "bible.n.01"
    },
    {
        "def": "a wheeled vehicle that has two wheels and is moved by foot pedals",
        "frequency": "f",
        "id": 95,
        "name": "bicycle",
        "synonyms": ["bicycle", "bike_(bicycle)"],
        "synset": "bicycle.n.01"
    },
    {
        "def": "a brim that projects to the front to shade the eyes",
        "frequency": "f",
        "id": 96,
        "name": "visor",
        "synonyms": ["visor", "vizor"],
        "synset": "bill.n.09"
    },
    {
        "def": "holds loose papers or magazines",
        "frequency": "c",
        "id": 97,
        "name": "binder",
        "synonyms": ["binder", "ring-binder"],
        "synset": "binder.n.03"
    },
    {
        "def": "an optical instrument designed for simultaneous use by both eyes",
        "frequency": "c",
        "id": 98,
        "name": "binoculars",
        "synonyms": ["binoculars", "field_glasses", "opera_glasses"],
        "synset": "binoculars.n.01"
    },
    {
        "def": "animal characterized by feathers and wings",
        "frequency": "f",
        "id": 99,
        "name": "bird",
        "synonyms": ["bird"],
        "synset": "bird.n.01"
    },
    {
        "def": "an outdoor device that supplies food for wild birds",
        "frequency": "r",
        "id": 100,
        "name": "birdfeeder",
        "synonyms": ["birdfeeder"],
        "synset": "bird_feeder.n.01"
    },
    {
        "def": "an ornamental basin (usually in a garden) for birds to bathe in",
        "frequency": "r",
        "id": 101,
        "name": "birdbath",
        "synonyms": ["birdbath"],
        "synset": "birdbath.n.01"
    },
    {
        "def": "a cage in which a bird can be kept",
        "frequency": "c",
        "id": 102,
        "name": "birdcage",
        "synonyms": ["birdcage"],
        "synset": "birdcage.n.01"
    },
    {
        "def": "a shelter for birds",
        "frequency": "c",
        "id": 103,
        "name": "birdhouse",
        "synonyms": ["birdhouse"],
        "synset": "birdhouse.n.01"
    },
    {
        "def": "decorated cake served at a birthday party",
        "frequency": "f",
        "id": 104,
        "name": "birthday_cake",
        "synonyms": ["birthday_cake"],
        "synset": "birthday_cake.n.01"
    },
    {
        "def": "a card expressing a birthday greeting",
        "frequency": "r",
        "id": 105,
        "name": "birthday_card",
        "synonyms": ["birthday_card"],
        "synset": "birthday_card.n.01"
    },
    {
        "def": "small round bread leavened with baking-powder or soda",
        "frequency": "r",
        "id": 106,
        "name": "biscuit_(bread)",
        "synonyms": ["biscuit_(bread)"],
        "synset": "biscuit.n.01"
    },
    {
        "def": "a flag usually bearing a white skull and crossbones on a black background",
        "frequency": "r",
        "id": 107,
        "name": "pirate_flag",
        "synonyms": ["pirate_flag"],
        "synset": "black_flag.n.01"
    },
    {
        "def": "sheep with a black coat",
        "frequency": "c",
        "id": 108,
        "name": "black_sheep",
        "synonyms": ["black_sheep"],
        "synset": "black_sheep.n.02"
    },
    {
        "def": "sheet of slate; for writing with chalk",
        "frequency": "c",
        "id": 109,
        "name": "blackboard",
        "synonyms": ["blackboard", "chalkboard"],
        "synset": "blackboard.n.01"
    },
    {
        "def": "bedding that keeps a person warm in bed",
        "frequency": "f",
        "id": 110,
        "name": "blanket",
        "synonyms": ["blanket"],
        "synset": "blanket.n.01"
    },
    {
        "def": "lightweight jacket; often striped in the colors of a club or school",
        "frequency": "c",
        "id": 111,
        "name": "blazer",
        "synonyms": ["blazer", "sport_jacket", "sport_coat", "sports_jacket", "sports_coat"],
        "synset": "blazer.n.01"
    },
    {
        "def": "an electrically powered mixer that mix or chop or liquefy foods",
        "frequency": "f",
        "id": 112,
        "name": "blender",
        "synonyms": ["blender", "liquidizer", "liquidiser"],
        "synset": "blender.n.01"
    },
    {
        "def": "a small nonrigid airship used for observation or as a barrage balloon",
        "frequency": "r",
        "id": 113,
        "name": "blimp",
        "synonyms": ["blimp"],
        "synset": "blimp.n.02"
    },
    {
        "def": "a light that flashes on and off; used as a signal or to send messages",
        "frequency": "c",
        "id": 114,
        "name": "blinker",
        "synonyms": ["blinker", "flasher"],
        "synset": "blinker.n.01"
    },
    {
        "def": "sweet edible dark-blue berries of blueberry plants",
        "frequency": "c",
        "id": 115,
        "name": "blueberry",
        "synonyms": ["blueberry"],
        "synset": "blueberry.n.02"
    },
    {
        "def": "an uncastrated male hog",
        "frequency": "r",
        "id": 116,
        "name": "boar",
        "synonyms": ["boar"],
        "synset": "boar.n.02"
    },
    {
        "def": "a flat portable surface (usually rectangular) designed for board games",
        "frequency": "r",
        "id": 117,
        "name": "gameboard",
        "synonyms": ["gameboard"],
        "synset": "board.n.09"
    },
    {
        "def": "a vessel for travel on water",
        "frequency": "f",
        "id": 118,
        "name": "boat",
        "synonyms": ["boat", "ship_(boat)"],
        "synset": "boat.n.01"
    },
    {
        "def": "a thing around which thread/tape/film or other flexible materials can be wound",
        "frequency": "c",
        "id": 119,
        "name": "bobbin",
        "synonyms": ["bobbin", "spool", "reel"],
        "synset": "bobbin.n.01"
    },
    {
        "def": "a flat wire hairpin used to hold bobbed hair in place",
        "frequency": "r",
        "id": 120,
        "name": "bobby_pin",
        "synonyms": ["bobby_pin", "hairgrip"],
        "synset": "bobby_pin.n.01"
    },
    {
        "def": "egg cooked briefly in the shell in gently boiling water",
        "frequency": "c",
        "id": 121,
        "name": "boiled_egg",
        "synonyms": ["boiled_egg", "coddled_egg"],
        "synset": "boiled_egg.n.01"
    },
    {
        "def": "a cord fastened around the neck with an ornamental clasp and worn as a necktie",
        "frequency": "r",
        "id": 122,
        "name": "bolo_tie",
        "synonyms": ["bolo_tie", "bolo", "bola_tie", "bola"],
        "synset": "bolo_tie.n.01"
    },
    {
        "def": "the part of a lock that is engaged or withdrawn with a key",
        "frequency": "c",
        "id": 123,
        "name": "deadbolt",
        "synonyms": ["deadbolt"],
        "synset": "bolt.n.03"
    },
    {
        "def": "a screw that screws into a nut to form a fastener",
        "frequency": "f",
        "id": 124,
        "name": "bolt",
        "synonyms": ["bolt"],
        "synset": "bolt.n.06"
    },
    {
        "def": "a hat tied under the chin",
        "frequency": "r",
        "id": 125,
        "name": "bonnet",
        "synonyms": ["bonnet"],
        "synset": "bonnet.n.01"
    },
    {
        "def": "a written work or composition that has been published",
        "frequency": "f",
        "id": 126,
        "name": "book",
        "synonyms": ["book"],
        "synset": "book.n.01"
    },
    {
        "def": "a bag in which students carry their books",
        "frequency": "r",
        "id": 127,
        "name": "book_bag",
        "synonyms": ["book_bag"],
        "synset": "book_bag.n.01"
    },
    {
        "def": "a piece of furniture with shelves for storing books",
        "frequency": "c",
        "id": 128,
        "name": "bookcase",
        "synonyms": ["bookcase"],
        "synset": "bookcase.n.01"
    },
    {
        "def": "a small book usually having a paper cover",
        "frequency": "c",
        "id": 129,
        "name": "booklet",
        "synonyms": ["booklet", "brochure", "leaflet", "pamphlet"],
        "synset": "booklet.n.01"
    },
    {
        "def": "a marker (a piece of paper or ribbon) placed between the pages of a book",
        "frequency": "r",
        "id": 130,
        "name": "bookmark",
        "synonyms": ["bookmark", "bookmarker"],
        "synset": "bookmark.n.01"
    },
    {
        "def": "a pole carrying an overhead microphone projected over a film or tv set",
        "frequency": "r",
        "id": 131,
        "name": "boom_microphone",
        "synonyms": ["boom_microphone", "microphone_boom"],
        "synset": "boom.n.04"
    },
    {
        "def": "footwear that covers the whole foot and lower leg",
        "frequency": "f",
        "id": 132,
        "name": "boot",
        "synonyms": ["boot"],
        "synset": "boot.n.01"
    },
    {
        "def": "a glass or plastic vessel used for storing drinks or other liquids",
        "frequency": "f",
        "id": 133,
        "name": "bottle",
        "synonyms": ["bottle"],
        "synset": "bottle.n.01"
    },
    {
        "def": "an opener for removing caps or corks from bottles",
        "frequency": "c",
        "id": 134,
        "name": "bottle_opener",
        "synonyms": ["bottle_opener"],
        "synset": "bottle_opener.n.01"
    },
    {
        "def": "an arrangement of flowers that is usually given as a present",
        "frequency": "c",
        "id": 135,
        "name": "bouquet",
        "synonyms": ["bouquet"],
        "synset": "bouquet.n.01"
    },
    {
        "def": "a weapon for shooting arrows",
        "frequency": "r",
        "id": 136,
        "name": "bow_(weapon)",
        "synonyms": ["bow_(weapon)"],
        "synset": "bow.n.04"
    },
    {
        "def": "a decorative interlacing of ribbons",
        "frequency": "f",
        "id": 137,
        "name": "bow_(decorative_ribbons)",
        "synonyms": ["bow_(decorative_ribbons)"],
        "synset": "bow.n.08"
    },
    {
        "def": "a man's tie that ties in a bow",
        "frequency": "f",
        "id": 138,
        "name": "bow-tie",
        "synonyms": ["bow-tie", "bowtie"],
        "synset": "bow_tie.n.01"
    },
    {
        "def": "a dish that is round and open at the top for serving foods",
        "frequency": "f",
        "id": 139,
        "name": "bowl",
        "synonyms": ["bowl"],
        "synset": "bowl.n.03"
    },
    {
        "def": "a small round container that is open at the top for holding tobacco",
        "frequency": "r",
        "id": 140,
        "name": "pipe_bowl",
        "synonyms": ["pipe_bowl"],
        "synset": "bowl.n.08"
    },
    {
        "def": "a felt hat that is round and hard with a narrow brim",
        "frequency": "c",
        "id": 141,
        "name": "bowler_hat",
        "synonyms": ["bowler_hat", "bowler", "derby_hat", "derby", "plug_hat"],
        "synset": "bowler_hat.n.01"
    },
    {
        "def": "a large ball with finger holes used in the sport of bowling",
        "frequency": "r",
        "id": 142,
        "name": "bowling_ball",
        "synonyms": ["bowling_ball"],
        "synset": "bowling_ball.n.01"
    },
    {
        "def": "a club-shaped wooden object used in bowling",
        "frequency": "r",
        "id": 143,
        "name": "bowling_pin",
        "synonyms": ["bowling_pin"],
        "synset": "bowling_pin.n.01"
    },
    {
        "def": "large glove coverings the fists of a fighter worn for the sport of boxing",
        "frequency": "r",
        "id": 144,
        "name": "boxing_glove",
        "synonyms": ["boxing_glove"],
        "synset": "boxing_glove.n.01"
    },
    {
        "def": "elastic straps that hold trousers up (usually used in the plural)",
        "frequency": "c",
        "id": 145,
        "name": "suspenders",
        "synonyms": ["suspenders"],
        "synset": "brace.n.06"
    },
    {
        "def": "jewelry worn around the wrist for decoration",
        "frequency": "f",
        "id": 146,
        "name": "bracelet",
        "synonyms": ["bracelet", "bangle"],
        "synset": "bracelet.n.02"
    },
    {
        "def": "a memorial made of brass",
        "frequency": "r",
        "id": 147,
        "name": "brass_plaque",
        "synonyms": ["brass_plaque"],
        "synset": "brass.n.07"
    },
    {
        "def": "an undergarment worn by women to support their breasts",
        "frequency": "c",
        "id": 148,
        "name": "brassiere",
        "synonyms": ["brassiere", "bra", "bandeau"],
        "synset": "brassiere.n.01"
    },
    {
        "def": "a container used to keep bread or cake in",
        "frequency": "c",
        "id": 149,
        "name": "bread-bin",
        "synonyms": ["bread-bin", "breadbox"],
        "synset": "bread-bin.n.01"
    },
    {
        "def": "a garment that provides covering for the loins",
        "frequency": "r",
        "id": 150,
        "name": "breechcloth",
        "synonyms": ["breechcloth", "breechclout", "loincloth"],
        "synset": "breechcloth.n.01"
    },
    {
        "def": "a gown worn by the bride at a wedding",
        "frequency": "c",
        "id": 151,
        "name": "bridal_gown",
        "synonyms": ["bridal_gown", "wedding_gown", "wedding_dress"],
        "synset": "bridal_gown.n.01"
    },
    {
        "def": "a case with a handle; for carrying papers or files or books",
        "frequency": "c",
        "id": 152,
        "name": "briefcase",
        "synonyms": ["briefcase"],
        "synset": "briefcase.n.01"
    },
    {
        "def": "a brush that is made with the short stiff hairs of an animal or plant",
        "frequency": "c",
        "id": 153,
        "name": "bristle_brush",
        "synonyms": ["bristle_brush"],
        "synset": "bristle_brush.n.01"
    },
    {
        "def": "plant with dense clusters of tight green flower buds",
        "frequency": "f",
        "id": 154,
        "name": "broccoli",
        "synonyms": ["broccoli"],
        "synset": "broccoli.n.01"
    },
    {
        "def": "a decorative pin worn by women",
        "frequency": "r",
        "id": 155,
        "name": "broach",
        "synonyms": ["broach"],
        "synset": "brooch.n.01"
    },
    {
        "def": "bundle of straws or twigs attached to a long handle; used for cleaning",
        "frequency": "c",
        "id": 156,
        "name": "broom",
        "synonyms": ["broom"],
        "synset": "broom.n.01"
    },
    {
        "def": "square or bar of very rich chocolate cake usually with nuts",
        "frequency": "c",
        "id": 157,
        "name": "brownie",
        "synonyms": ["brownie"],
        "synset": "brownie.n.03"
    },
    {
        "def": "the small edible cabbage-like buds growing along a stalk",
        "frequency": "c",
        "id": 158,
        "name": "brussels_sprouts",
        "synonyms": ["brussels_sprouts"],
        "synset": "brussels_sprouts.n.01"
    },
    {
        "def": "a kind of chewing gum that can be blown into bubbles",
        "frequency": "r",
        "id": 159,
        "name": "bubble_gum",
        "synonyms": ["bubble_gum"],
        "synset": "bubble_gum.n.01"
    },
    {
        "def": "a roughly cylindrical vessel that is open at the top",
        "frequency": "f",
        "id": 160,
        "name": "bucket",
        "synonyms": ["bucket", "pail"],
        "synset": "bucket.n.01"
    },
    {
        "def": "a small lightweight carriage; drawn by a single horse",
        "frequency": "r",
        "id": 161,
        "name": "horse_buggy",
        "synonyms": ["horse_buggy"],
        "synset": "buggy.n.01"
    },
    {
        "def": "mature male cow",
        "frequency": "c",
        "id": 162,
        "name": "bull",
        "synonyms": ["bull"],
        "synset": "bull.n.11"
    },
    {
        "def": "a thickset short-haired dog with a large head and strong undershot lower jaw",
        "frequency": "r",
        "id": 163,
        "name": "bulldog",
        "synonyms": ["bulldog"],
        "synset": "bulldog.n.01"
    },
    {
        "def": "large powerful tractor; a large blade in front flattens areas of ground",
        "frequency": "r",
        "id": 164,
        "name": "bulldozer",
        "synonyms": ["bulldozer", "dozer"],
        "synset": "bulldozer.n.01"
    },
    {
        "def": "a high-speed passenger train",
        "frequency": "c",
        "id": 165,
        "name": "bullet_train",
        "synonyms": ["bullet_train"],
        "synset": "bullet_train.n.01"
    },
    {
        "def": "a board that hangs on a wall; displays announcements",
        "frequency": "c",
        "id": 166,
        "name": "bulletin_board",
        "synonyms": ["bulletin_board", "notice_board"],
        "synset": "bulletin_board.n.02"
    },
    {
        "def": "a vest capable of resisting the impact of a bullet",
        "frequency": "r",
        "id": 167,
        "name": "bulletproof_vest",
        "synonyms": ["bulletproof_vest"],
        "synset": "bulletproof_vest.n.01"
    },
    {
        "def": "a portable loudspeaker with built-in microphone and amplifier",
        "frequency": "c",
        "id": 168,
        "name": "bullhorn",
        "synonyms": ["bullhorn", "megaphone"],
        "synset": "bullhorn.n.01"
    },
    {
        "def": "beef cured or pickled in brine",
        "frequency": "r",
        "id": 169,
        "name": "corned_beef",
        "synonyms": ["corned_beef", "corn_beef"],
        "synset": "bully_beef.n.01"
    },
    {
        "def": "small rounded bread either plain or sweet",
        "frequency": "f",
        "id": 170,
        "name": "bun",
        "synonyms": ["bun", "roll"],
        "synset": "bun.n.01"
    },
    {
        "def": "beds built one above the other",
        "frequency": "c",
        "id": 171,
        "name": "bunk_bed",
        "synonyms": ["bunk_bed"],
        "synset": "bunk_bed.n.01"
    },
    {
        "def": "a float attached by rope to the seabed to mark channels in a harbor or underwater "
               "hazards",
        "frequency": "f",
        "id": 172,
        "name": "buoy",
        "synonyms": ["buoy"],
        "synset": "buoy.n.01"
    },
    {
        "def": "a flour tortilla folded around a filling",
        "frequency": "r",
        "id": 173,
        "name": "burrito",
        "synonyms": ["burrito"],
        "synset": "burrito.n.01"
    },
    {
        "def": "a vehicle carrying many passengers; used for public transport",
        "frequency": "f",
        "id": 174,
        "name": "bus_(vehicle)",
        "synonyms": ["bus_(vehicle)", "autobus", "charabanc", "double-decker", "motorbus",
                     "motorcoach"],
        "synset": "bus.n.01"
    },
    {
        "def": "a card on which are printed the person's name and business affiliation",
        "frequency": "c",
        "id": 175,
        "name": "business_card",
        "synonyms": ["business_card"],
        "synset": "business_card.n.01"
    },
    {
        "def": "a large sharp knife for cutting or trimming meat",
        "frequency": "c",
        "id": 176,
        "name": "butcher_knife",
        "synonyms": ["butcher_knife"],
        "synset": "butcher_knife.n.01"
    },
    {
        "def": "an edible emulsion of fat globules made by churning milk or cream; for cooking and "
               "table use",
        "frequency": "c",
        "id": 177,
        "name": "butter",
        "synonyms": ["butter"],
        "synset": "butter.n.01"
    },
    {
        "def": "insect typically having a slender body with knobbed antennae and broad colorful "
               "wings",
        "frequency": "c",
        "id": 178,
        "name": "butterfly",
        "synonyms": ["butterfly"],
        "synset": "butterfly.n.01"
    },
    {
        "def": "a round fastener sewn to shirts and coats etc to fit through buttonholes",
        "frequency": "f",
        "id": 179,
        "name": "button",
        "synonyms": ["button"],
        "synset": "button.n.01"
    },
    {
        "def": "a car that takes passengers where they want to go in exchange for money",
        "frequency": "f",
        "id": 180,
        "name": "cab_(taxi)",
        "synonyms": ["cab_(taxi)", "taxi", "taxicab"],
        "synset": "cab.n.03"
    },
    {
        "def": "a small tent used as a dressing room beside the sea or a swimming pool",
        "frequency": "r",
        "id": 181,
        "name": "cabana",
        "synonyms": ["cabana"],
        "synset": "cabana.n.01"
    },
    {
        "def": "a car on a freight train for use of the train crew; usually the last car on the "
               "train",
        "frequency": "r",
        "id": 182,
        "name": "cabin_car",
        "synonyms": ["cabin_car", "caboose"],
        "synset": "cabin_car.n.01"
    },
    {
        "def": "a piece of furniture resembling a cupboard with doors and shelves and drawers",
        "frequency": "f",
        "id": 183,
        "name": "cabinet",
        "synonyms": ["cabinet"],
        "synset": "cabinet.n.01"
    },
    {
        "def": "a storage compartment for clothes and valuables; usually it has a lock",
        "frequency": "r",
        "id": 184,
        "name": "locker",
        "synonyms": ["locker", "storage_locker"],
        "synset": "cabinet.n.03"
    },
    {
        "def": "baked goods made from or based on a mixture of flour, sugar, eggs, and fat",
        "frequency": "f",
        "id": 185,
        "name": "cake",
        "synonyms": ["cake"],
        "synset": "cake.n.03"
    },
    {
        "def": "a small machine that is used for mathematical calculations",
        "frequency": "c",
        "id": 186,
        "name": "calculator",
        "synonyms": ["calculator"],
        "synset": "calculator.n.02"
    },
    {
        "def": "a list or register of events (appointments/social events/court cases, etc)",
        "frequency": "f",
        "id": 187,
        "name": "calendar",
        "synonyms": ["calendar"],
        "synset": "calendar.n.02"
    },
    {
        "def": "young of domestic cattle",
        "frequency": "c",
        "id": 188,
        "name": "calf",
        "synonyms": ["calf"],
        "synset": "calf.n.01"
    },
    {
        "def": "a portable television camera and videocassette recorder",
        "frequency": "c",
        "id": 189,
        "name": "camcorder",
        "synonyms": ["camcorder"],
        "synset": "camcorder.n.01"
    },
    {
        "def": "cud-chewing mammal used as a draft or saddle animal in desert regions",
        "frequency": "c",
        "id": 190,
        "name": "camel",
        "synonyms": ["camel"],
        "synset": "camel.n.01"
    },
    {
        "def": "equipment for taking photographs",
        "frequency": "f",
        "id": 191,
        "name": "camera",
        "synonyms": ["camera"],
        "synset": "camera.n.01"
    },
    {
        "def": "a lens that focuses the image in a camera",
        "frequency": "c",
        "id": 192,
        "name": "camera_lens",
        "synonyms": ["camera_lens"],
        "synset": "camera_lens.n.01"
    },
    {
        "def": "a recreational vehicle equipped for camping out while traveling",
        "frequency": "c",
        "id": 193,
        "name": "camper_(vehicle)",
        "synonyms": ["camper_(vehicle)", "camping_bus", "motor_home"],
        "synset": "camper.n.02"
    },
    {
        "def": "airtight sealed metal container for food or drink or paint etc.",
        "frequency": "f",
        "id": 194,
        "name": "can",
        "synonyms": ["can", "tin_can"],
        "synset": "can.n.01"
    },
    {
        "def": "a device for cutting cans open",
        "frequency": "c",
        "id": 195,
        "name": "can_opener",
        "synonyms": ["can_opener", "tin_opener"],
        "synset": "can_opener.n.01"
    },
    {
        "def": "branched candlestick; ornamental; has several lights",
        "frequency": "r",
        "id": 196,
        "name": "candelabrum",
        "synonyms": ["candelabrum", "candelabra"],
        "synset": "candelabrum.n.01"
    },
    {
        "def": "stick of wax with a wick in the middle",
        "frequency": "f",
        "id": 197,
        "name": "candle",
        "synonyms": ["candle", "candlestick"],
        "synset": "candle.n.01"
    },
    {
        "def": "a holder with sockets for candles",
        "frequency": "f",
        "id": 198,
        "name": "candle_holder",
        "synonyms": ["candle_holder"],
        "synset": "candlestick.n.01"
    },
    {
        "def": "a candy shaped as a bar",
        "frequency": "r",
        "id": 199,
        "name": "candy_bar",
        "synonyms": ["candy_bar"],
        "synset": "candy_bar.n.01"
    },
    {
        "def": "a hard candy in the shape of a rod (usually with stripes)",
        "frequency": "c",
        "id": 200,
        "name": "candy_cane",
        "synonyms": ["candy_cane"],
        "synset": "candy_cane.n.01"
    },
    {
        "def": "a stick that people can lean on to help them walk",
        "frequency": "c",
        "id": 201,
        "name": "walking_cane",
        "synonyms": ["walking_cane"],
        "synset": "cane.n.01"
    },
    {
        "def": "metal container for storing dry foods such as tea or flour",
        "frequency": "c",
        "id": 202,
        "name": "canister",
        "synonyms": ["canister", "cannister"],
        "synset": "canister.n.02"
    },
    {
        "def": "heavy gun fired from a tank",
        "frequency": "r",
        "id": 203,
        "name": "cannon",
        "synonyms": ["cannon"],
        "synset": "cannon.n.02"
    },
    {
        "def": "small and light boat; pointed at both ends; propelled with a paddle",
        "frequency": "c",
        "id": 204,
        "name": "canoe",
        "synonyms": ["canoe"],
        "synset": "canoe.n.01"
    },
    {
        "def": "the fruit of a cantaloup vine; small to medium-sized melon with yellowish flesh",
        "frequency": "r",
        "id": 205,
        "name": "cantaloup",
        "synonyms": ["cantaloup", "cantaloupe"],
        "synset": "cantaloup.n.02"
    },
    {
        "def": "a flask for carrying water; used by soldiers or travelers",
        "frequency": "r",
        "id": 206,
        "name": "canteen",
        "synonyms": ["canteen"],
        "synset": "canteen.n.01"
    },
    {
        "def": "a tight-fitting headwear",
        "frequency": "c",
        "id": 207,
        "name": "cap_(headwear)",
        "synonyms": ["cap_(headwear)"],
        "synset": "cap.n.01"
    },
    {
        "def": "a top (as for a bottle)",
        "frequency": "f",
        "id": 208,
        "name": "bottle_cap",
        "synonyms": ["bottle_cap", "cap_(container_lid)"],
        "synset": "cap.n.02"
    },
    {
        "def": "a sleeveless garment like a cloak but shorter",
        "frequency": "r",
        "id": 209,
        "name": "cape",
        "synonyms": ["cape"],
        "synset": "cape.n.02"
    },
    {
        "def": "equal parts of espresso and steamed milk",
        "frequency": "c",
        "id": 210,
        "name": "cappuccino",
        "synonyms": ["cappuccino", "coffee_cappuccino"],
        "synset": "cappuccino.n.01"
    },
    {
        "def": "a motor vehicle with four wheels",
        "frequency": "f",
        "id": 211,
        "name": "car_(automobile)",
        "synonyms": ["car_(automobile)", "auto_(automobile)", "automobile"],
        "synset": "car.n.01"
    },
    {
        "def": "a wheeled vehicle adapted to the rails of railroad",
        "frequency": "f",
        "id": 212,
        "name": "railcar_(part_of_a_train)",
        "synonyms": ["railcar_(part_of_a_train)", "railway_car_(part_of_a_train)",
                     "railroad_car_(part_of_a_train)"],
        "synset": "car.n.02"
    },
    {
        "def": "where passengers ride up and down",
        "frequency": "r",
        "id": 213,
        "name": "elevator_car",
        "synonyms": ["elevator_car"],
        "synset": "car.n.04"
    },
    {
        "def": "a battery in a motor vehicle",
        "frequency": "r",
        "id": 214,
        "name": "car_battery",
        "synonyms": ["car_battery", "automobile_battery"],
        "synset": "car_battery.n.01"
    },
    {
        "def": "a card certifying the identity of the bearer",
        "frequency": "c",
        "id": 215,
        "name": "identity_card",
        "synonyms": ["identity_card"],
        "synset": "card.n.02"
    },
    {
        "def": "a rectangular piece of paper used to send messages (e.g. greetings or pictures)",
        "frequency": "c",
        "id": 216,
        "name": "card",
        "synonyms": ["card"],
        "synset": "card.n.03"
    },
    {
        "def": "knitted jacket that is fastened up the front with buttons or a zipper",
        "frequency": "r",
        "id": 217,
        "name": "cardigan",
        "synonyms": ["cardigan"],
        "synset": "cardigan.n.01"
    },
    {
        "def": "a ship designed to carry cargo",
        "frequency": "r",
        "id": 218,
        "name": "cargo_ship",
        "synonyms": ["cargo_ship", "cargo_vessel"],
        "synset": "cargo_ship.n.01"
    },
    {
        "def": "plant with pink to purple-red spice-scented usually double flowers",
        "frequency": "r",
        "id": 219,
        "name": "carnation",
        "synonyms": ["carnation"],
        "synset": "carnation.n.01"
    },
    {
        "def": "a vehicle with wheels drawn by one or more horses",
        "frequency": "c",
        "id": 220,
        "name": "horse_carriage",
        "synonyms": ["horse_carriage"],
        "synset": "carriage.n.02"
    },
    {
        "def": "deep orange edible root of the cultivated carrot plant",
        "frequency": "f",
        "id": 221,
        "name": "carrot",
        "synonyms": ["carrot"],
        "synset": "carrot.n.01"
    },
    {
        "def": "a capacious bag or basket",
        "frequency": "c",
        "id": 222,
        "name": "tote_bag",
        "synonyms": ["tote_bag"],
        "synset": "carryall.n.01"
    },
    {
        "def": "a heavy open wagon usually having two wheels and drawn by an animal",
        "frequency": "c",
        "id": 223,
        "name": "cart",
        "synonyms": ["cart"],
        "synset": "cart.n.01"
    },
    {
        "def": "a box made of cardboard; opens by flaps on top",
        "frequency": "c",
        "id": 224,
        "name": "carton",
        "synonyms": ["carton"],
        "synset": "carton.n.02"
    },
    {
        "def": "a cashbox with an adding machine to register transactions",
        "frequency": "c",
        "id": 225,
        "name": "cash_register",
        "synonyms": ["cash_register", "register_(for_cash_transactions)"],
        "synset": "cash_register.n.01"
    },
    {
        "def": "food cooked and served in a casserole",
        "frequency": "r",
        "id": 226,
        "name": "casserole",
        "synonyms": ["casserole"],
        "synset": "casserole.n.01"
    },
    {
        "def": "a container that holds a magnetic tape used for recording or playing sound or "
               "video",
        "frequency": "r",
        "id": 227,
        "name": "cassette",
        "synonyms": ["cassette"],
        "synset": "cassette.n.01"
    },
    {
        "def": "bandage consisting of a firm covering that immobilizes broken bones while they "
               "heal",
        "frequency": "c",
        "id": 228,
        "name": "cast",
        "synonyms": ["cast", "plaster_cast", "plaster_bandage"],
        "synset": "cast.n.05"
    },
    {
        "def": "a domestic house cat",
        "frequency": "f",
        "id": 229,
        "name": "cat",
        "synonyms": ["cat"],
        "synset": "cat.n.01"
    },
    {
        "def": "edible compact head of white undeveloped flowers",
        "frequency": "c",
        "id": 230,
        "name": "cauliflower",
        "synonyms": ["cauliflower"],
        "synset": "cauliflower.n.02"
    },
    {
        "def": "salted roe of sturgeon or other large fish; usually served as an hors d'oeuvre",
        "frequency": "r",
        "id": 231,
        "name": "caviar",
        "synonyms": ["caviar", "caviare"],
        "synset": "caviar.n.01"
    },
    {
        "def": "ground pods and seeds of pungent red peppers of the genus Capsicum",
        "frequency": "c",
        "id": 232,
        "name": "cayenne_(spice)",
        "synonyms": ["cayenne_(spice)", "cayenne_pepper_(spice)", "red_pepper_(spice)"],
        "synset": "cayenne.n.02"
    },
    {
        "def": "electronic equipment for playing compact discs (CDs)",
        "frequency": "c",
        "id": 233,
        "name": "CD_player",
        "synonyms": ["CD_player"],
        "synset": "cd_player.n.01"
    },
    {
        "def": "widely cultivated herb with aromatic leaf stalks that are eaten raw or cooked",
        "frequency": "c",
        "id": 234,
        "name": "celery",
        "synonyms": ["celery"],
        "synset": "celery.n.01"
    },
    {
        "def": "a hand-held mobile telephone",
        "frequency": "f",
        "id": 235,
        "name": "cellular_telephone",
        "synonyms": ["cellular_telephone", "cellular_phone", "cellphone", "mobile_phone",
                     "smart_phone"],
        "synset": "cellular_telephone.n.01"
    },
    {
        "def": "(Middle Ages) flexible armor made of interlinked metal rings",
        "frequency": "r",
        "id": 236,
        "name": "chain_mail",
        "synonyms": ["chain_mail", "ring_mail", "chain_armor", "chain_armour", "ring_armor",
                     "ring_armour"],
        "synset": "chain_mail.n.01"
    },
    {
        "def": "a seat for one person, with a support for the back",
        "frequency": "f",
        "id": 237,
        "name": "chair",
        "synonyms": ["chair"],
        "synset": "chair.n.01"
    },
    {
        "def": "a long chair; for reclining",
        "frequency": "r",
        "id": 238,
        "name": "chaise_longue",
        "synonyms": ["chaise_longue", "chaise", "daybed"],
        "synset": "chaise_longue.n.01"
    },
    {
        "def": "a white sparkling wine produced in Champagne or resembling that produced there",
        "frequency": "r",
        "id": 239,
        "name": "champagne",
        "synonyms": ["champagne"],
        "synset": "champagne.n.01"
    },
    {
        "def": "branched lighting fixture; often ornate; hangs from the ceiling",
        "frequency": "f",
        "id": 240,
        "name": "chandelier",
        "synonyms": ["chandelier"],
        "synset": "chandelier.n.01"
    },
    {
        "def": "leather leggings without a seat; worn over trousers by cowboys to protect their "
               "legs",
        "frequency": "r",
        "id": 241,
        "name": "chap",
        "synonyms": ["chap"],
        "synset": "chap.n.04"
    },
    {
        "def": "a book issued to holders of checking accounts",
        "frequency": "r",
        "id": 242,
        "name": "checkbook",
        "synonyms": ["checkbook", "chequebook"],
        "synset": "checkbook.n.01"
    },
    {
        "def": "a board having 64 squares of two alternating colors",
        "frequency": "r",
        "id": 243,
        "name": "checkerboard",
        "synonyms": ["checkerboard"],
        "synset": "checkerboard.n.01"
    },
    {
        "def": "a red fruit with a single hard stone",
        "frequency": "c",
        "id": 244,
        "name": "cherry",
        "synonyms": ["cherry"],
        "synset": "cherry.n.03"
    },
    {
        "def": "a checkerboard used to play chess",
        "frequency": "r",
        "id": 245,
        "name": "chessboard",
        "synonyms": ["chessboard"],
        "synset": "chessboard.n.01"
    },
    {
        "def": "furniture with drawers for keeping clothes",
        "frequency": "r",
        "id": 246,
        "name": "chest_of_drawers_(furniture)",
        "synonyms": ["chest_of_drawers_(furniture)", "bureau_(furniture)", "chest_(furniture)"],
        "synset": "chest_of_drawers.n.01"
    },
    {
        "def": "a domestic fowl bred for flesh or eggs",
        "frequency": "c",
        "id": 247,
        "name": "chicken_(animal)",
        "synonyms": ["chicken_(animal)"],
        "synset": "chicken.n.02"
    },
    {
        "def": "a galvanized wire network with a hexagonal mesh; used to build fences",
        "frequency": "c",
        "id": 248,
        "name": "chicken_wire",
        "synonyms": ["chicken_wire"],
        "synset": "chicken_wire.n.01"
    },
    {
        "def": "the seed of the chickpea plant; usually dried",
        "frequency": "r",
        "id": 249,
        "name": "chickpea",
        "synonyms": ["chickpea", "garbanzo"],
        "synset": "chickpea.n.01"
    },
    {
        "def": "an old breed of tiny short-haired dog with protruding eyes from Mexico",
        "frequency": "r",
        "id": 250,
        "name": "Chihuahua",
        "synonyms": ["Chihuahua"],
        "synset": "chihuahua.n.03"
    },
    {
        "def": "very hot and finely tapering pepper of special pungency",
        "frequency": "r",
        "id": 251,
        "name": "chili_(vegetable)",
        "synonyms": ["chili_(vegetable)", "chili_pepper_(vegetable)", "chilli_(vegetable)",
                     "chilly_(vegetable)", "chile_(vegetable)"],
        "synset": "chili.n.02"
    },
    {
        "def": "an instrument consisting of a set of bells that are struck with a hammer",
        "frequency": "r",
        "id": 252,
        "name": "chime",
        "synonyms": ["chime", "gong"],
        "synset": "chime.n.01"
    },
    {
        "def": "dishware made of high quality porcelain",
        "frequency": "r",
        "id": 253,
        "name": "chinaware",
        "synonyms": ["chinaware"],
        "synset": "chinaware.n.01"
    },
    {
        "def": "a thin crisp slice of potato fried in deep fat",
        "frequency": "c",
        "id": 254,
        "name": "crisp_(potato_chip)",
        "synonyms": ["crisp_(potato_chip)", "potato_chip"],
        "synset": "chip.n.04"
    },
    {
        "def": "a small disk-shaped counter used to represent money when gambling",
        "frequency": "r",
        "id": 255,
        "name": "poker_chip",
        "synonyms": ["poker_chip"],
        "synset": "chip.n.06"
    },
    {
        "def": "a bar of chocolate candy",
        "frequency": "c",
        "id": 256,
        "name": "chocolate_bar",
        "synonyms": ["chocolate_bar"],
        "synset": "chocolate_bar.n.01"
    },
    {
        "def": "cake containing chocolate",
        "frequency": "c",
        "id": 257,
        "name": "chocolate_cake",
        "synonyms": ["chocolate_cake"],
        "synset": "chocolate_cake.n.01"
    },
    {
        "def": "milk flavored with chocolate syrup",
        "frequency": "r",
        "id": 258,
        "name": "chocolate_milk",
        "synonyms": ["chocolate_milk"],
        "synset": "chocolate_milk.n.01"
    },
    {
        "def": "dessert mousse made with chocolate",
        "frequency": "r",
        "id": 259,
        "name": "chocolate_mousse",
        "synonyms": ["chocolate_mousse"],
        "synset": "chocolate_mousse.n.01"
    },
    {
        "def": "necklace that fits tightly around the neck",
        "frequency": "f",
        "id": 260,
        "name": "choker",
        "synonyms": ["choker", "collar", "neckband"],
        "synset": "choker.n.03"
    },
    {
        "def": "a wooden board where meats or vegetables can be cut",
        "frequency": "f",
        "id": 261,
        "name": "chopping_board",
        "synonyms": ["chopping_board", "cutting_board", "chopping_block"],
        "synset": "chopping_board.n.01"
    },
    {
        "def": "one of a pair of slender sticks used as oriental tableware to eat food with",
        "frequency": "c",
        "id": 262,
        "name": "chopstick",
        "synonyms": ["chopstick"],
        "synset": "chopstick.n.01"
    },
    {
        "def": "an ornamented evergreen used as a Christmas decoration",
        "frequency": "f",
        "id": 263,
        "name": "Christmas_tree",
        "synonyms": ["Christmas_tree"],
        "synset": "christmas_tree.n.05"
    },
    {
        "def": "sloping channel through which things can descend",
        "frequency": "c",
        "id": 264,
        "name": "slide",
        "synonyms": ["slide"],
        "synset": "chute.n.02"
    },
    {
        "def": "a beverage made from juice pressed from apples",
        "frequency": "r",
        "id": 265,
        "name": "cider",
        "synonyms": ["cider", "cyder"],
        "synset": "cider.n.01"
    },
    {
        "def": "a box for holding cigars",
        "frequency": "r",
        "id": 266,
        "name": "cigar_box",
        "synonyms": ["cigar_box"],
        "synset": "cigar_box.n.01"
    },
    {
        "def": "finely ground tobacco wrapped in paper; for smoking",
        "frequency": "c",
        "id": 267,
        "name": "cigarette",
        "synonyms": ["cigarette"],
        "synset": "cigarette.n.01"
    },
    {
        "def": "a small flat case for holding cigarettes",
        "frequency": "c",
        "id": 268,
        "name": "cigarette_case",
        "synonyms": ["cigarette_case", "cigarette_pack"],
        "synset": "cigarette_case.n.01"
    },
    {
        "def": "a tank that holds the water used to flush a toilet",
        "frequency": "f",
        "id": 269,
        "name": "cistern",
        "synonyms": ["cistern", "water_tank"],
        "synset": "cistern.n.02"
    },
    {
        "def": "a single-reed instrument with a straight tube",
        "frequency": "r",
        "id": 270,
        "name": "clarinet",
        "synonyms": ["clarinet"],
        "synset": "clarinet.n.01"
    },
    {
        "def": "a fastener (as a buckle or hook) that is used to hold two things together",
        "frequency": "r",
        "id": 271,
        "name": "clasp",
        "synonyms": ["clasp"],
        "synset": "clasp.n.01"
    },
    {
        "def": "a preparation used in cleaning something",
        "frequency": "c",
        "id": 272,
        "name": "cleansing_agent",
        "synonyms": ["cleansing_agent", "cleanser", "cleaner"],
        "synset": "cleansing_agent.n.01"
    },
    {
        "def": "a variety of mandarin orange",
        "frequency": "r",
        "id": 273,
        "name": "clementine",
        "synonyms": ["clementine"],
        "synset": "clementine.n.01"
    },
    {
        "def": "any of various small fasteners used to hold loose articles together",
        "frequency": "c",
        "id": 274,
        "name": "clip",
        "synonyms": ["clip"],
        "synset": "clip.n.03"
    },
    {
        "def": "a small writing board with a clip at the top for holding papers",
        "frequency": "c",
        "id": 275,
        "name": "clipboard",
        "synonyms": ["clipboard"],
        "synset": "clipboard.n.01"
    },
    {
        "def": "a timepiece that shows the time of day",
        "frequency": "f",
        "id": 276,
        "name": "clock",
        "synonyms": ["clock", "timepiece", "timekeeper"],
        "synset": "clock.n.01"
    },
    {
        "def": "a tower with a large clock visible high up on an outside face",
        "frequency": "f",
        "id": 277,
        "name": "clock_tower",
        "synonyms": ["clock_tower"],
        "synset": "clock_tower.n.01"
    },
    {
        "def": "a hamper that holds dirty clothes to be washed or wet clothes to be dried",
        "frequency": "c",
        "id": 278,
        "name": "clothes_hamper",
        "synonyms": ["clothes_hamper", "laundry_basket", "clothes_basket"],
        "synset": "clothes_hamper.n.01"
    },
    {
        "def": "wood or plastic fastener; for holding clothes on a clothesline",
        "frequency": "c",
        "id": 279,
        "name": "clothespin",
        "synonyms": ["clothespin", "clothes_peg"],
        "synset": "clothespin.n.01"
    },
    {
        "def": "a woman's strapless purse that is carried in the hand",
        "frequency": "r",
        "id": 280,
        "name": "clutch_bag",
        "synonyms": ["clutch_bag"],
        "synset": "clutch_bag.n.01"
    },
    {
        "def": "a covering (plate or mat) that protects the surface of a table",
        "frequency": "f",
        "id": 281,
        "name": "coaster",
        "synonyms": ["coaster"],
        "synset": "coaster.n.03"
    },
    {
        "def": "an outer garment that has sleeves and covers the body from shoulder down",
        "frequency": "f",
        "id": 282,
        "name": "coat",
        "synonyms": ["coat"],
        "synset": "coat.n.01"
    },
    {
        "def": "a hanger that is shaped like a person's shoulders",
        "frequency": "c",
        "id": 283,
        "name": "coat_hanger",
        "synonyms": ["coat_hanger", "clothes_hanger", "dress_hanger"],
        "synset": "coat_hanger.n.01"
    },
    {
        "def": "a rack with hooks for temporarily holding coats and hats",
        "frequency": "r",
        "id": 284,
        "name": "coatrack",
        "synonyms": ["coatrack", "hatrack"],
        "synset": "coatrack.n.01"
    },
    {
        "def": "adult male chicken",
        "frequency": "c",
        "id": 285,
        "name": "cock",
        "synonyms": ["cock", "rooster"],
        "synset": "cock.n.04"
    },
    {
        "def": "large hard-shelled brown oval nut with a fibrous husk",
        "frequency": "c",
        "id": 286,
        "name": "coconut",
        "synonyms": ["coconut", "cocoanut"],
        "synset": "coconut.n.02"
    },
    {
        "def": "filter (usually of paper) that passes the coffee and retains the coffee grounds",
        "frequency": "r",
        "id": 287,
        "name": "coffee_filter",
        "synonyms": ["coffee_filter"],
        "synset": "coffee_filter.n.01"
    },
    {
        "def": "a kitchen appliance for brewing coffee automatically",
        "frequency": "f",
        "id": 288,
        "name": "coffee_maker",
        "synonyms": ["coffee_maker", "coffee_machine"],
        "synset": "coffee_maker.n.01"
    },
    {
        "def": "low table where magazines can be placed and coffee or cocktails are served",
        "frequency": "f",
        "id": 289,
        "name": "coffee_table",
        "synonyms": ["coffee_table", "cocktail_table"],
        "synset": "coffee_table.n.01"
    },
    {
        "def": "tall pot in which coffee is brewed",
        "frequency": "c",
        "id": 290,
        "name": "coffeepot",
        "synonyms": ["coffeepot"],
        "synset": "coffeepot.n.01"
    },
    {
        "def": "tubing that is wound in a spiral",
        "frequency": "r",
        "id": 291,
        "name": "coil",
        "synonyms": ["coil"],
        "synset": "coil.n.05"
    },
    {
        "def": "a flat metal piece (usually a disc) used as money",
        "frequency": "c",
        "id": 292,
        "name": "coin",
        "synonyms": ["coin"],
        "synset": "coin.n.01"
    },
    {
        "def": "bowl-shaped strainer; used to wash or drain foods",
        "frequency": "r",
        "id": 293,
        "name": "colander",
        "synonyms": ["colander", "cullender"],
        "synset": "colander.n.01"
    },
    {
        "def": "basically shredded cabbage",
        "frequency": "c",
        "id": 294,
        "name": "coleslaw",
        "synonyms": ["coleslaw", "slaw"],
        "synset": "coleslaw.n.01"
    },
    {
        "def": "any material used for its color",
        "frequency": "r",
        "id": 295,
        "name": "coloring_material",
        "synonyms": ["coloring_material", "colouring_material"],
        "synset": "coloring_material.n.01"
    },
    {
        "def": "lock that can be opened only by turning dials in a special sequence",
        "frequency": "r",
        "id": 296,
        "name": "combination_lock",
        "synonyms": ["combination_lock"],
        "synset": "combination_lock.n.01"
    },
    {
        "def": "device used for an infant to suck or bite on",
        "frequency": "c",
        "id": 297,
        "name": "pacifier",
        "synonyms": ["pacifier", "teething_ring"],
        "synset": "comforter.n.04"
    },
    {
        "def": "a magazine devoted to comic strips",
        "frequency": "r",
        "id": 298,
        "name": "comic_book",
        "synonyms": ["comic_book"],
        "synset": "comic_book.n.01"
    },
    {
        "def": "a keyboard that is a data input device for computers",
        "frequency": "f",
        "id": 299,
        "name": "computer_keyboard",
        "synonyms": ["computer_keyboard", "keyboard_(computer)"],
        "synset": "computer_keyboard.n.01"
    },
    {
        "def": "a machine with a large revolving drum in which cement/concrete is mixed",
        "frequency": "r",
        "id": 300,
        "name": "concrete_mixer",
        "synonyms": ["concrete_mixer", "cement_mixer"],
        "synset": "concrete_mixer.n.01"
    },
    {
        "def": "a cone-shaped object used to direct traffic",
        "frequency": "f",
        "id": 301,
        "name": "cone",
        "synonyms": ["cone", "traffic_cone"],
        "synset": "cone.n.01"
    },
    {
        "def": "a mechanism that controls the operation of a machine",
        "frequency": "f",
        "id": 302,
        "name": "control",
        "synonyms": ["control", "controller"],
        "synset": "control.n.09"
    },
    {
        "def": "a car that has top that can be folded or removed",
        "frequency": "r",
        "id": 303,
        "name": "convertible_(automobile)",
        "synonyms": ["convertible_(automobile)"],
        "synset": "convertible.n.01"
    },
    {
        "def": "a sofa that can be converted into a bed",
        "frequency": "r",
        "id": 304,
        "name": "sofa_bed",
        "synonyms": ["sofa_bed"],
        "synset": "convertible.n.03"
    },
    {
        "def": "any of various small flat sweet cakes (`biscuit' is the British term)",
        "frequency": "c",
        "id": 305,
        "name": "cookie",
        "synonyms": ["cookie", "cooky", "biscuit_(cookie)"],
        "synset": "cookie.n.01"
    },
    {
        "def": "a jar in which cookies are kept (and sometimes money is hidden)",
        "frequency": "r",
        "id": 306,
        "name": "cookie_jar",
        "synonyms": ["cookie_jar", "cooky_jar"],
        "synset": "cookie_jar.n.01"
    },
    {
        "def": "a kitchen utensil made of material that does not melt easily; used for cooking",
        "frequency": "r",
        "id": 307,
        "name": "cooking_utensil",
        "synonyms": ["cooking_utensil"],
        "synset": "cooking_utensil.n.01"
    },
    {
        "def": "an insulated box for storing food often with ice",
        "frequency": "f",
        "id": 308,
        "name": "cooler_(for_food)",
        "synonyms": ["cooler_(for_food)", "ice_chest"],
        "synset": "cooler.n.01"
    },
    {
        "def": "the plug in the mouth of a bottle (especially a wine bottle)",
        "frequency": "c",
        "id": 309,
        "name": "cork_(bottle_plug)",
        "synonyms": ["cork_(bottle_plug)", "bottle_cork"],
        "synset": "cork.n.04"
    },
    {
        "def": "a sheet consisting of cork granules",
        "frequency": "r",
        "id": 310,
        "name": "corkboard",
        "synonyms": ["corkboard"],
        "synset": "corkboard.n.01"
    },
    {
        "def": "a bottle opener that pulls corks",
        "frequency": "r",
        "id": 311,
        "name": "corkscrew",
        "synonyms": ["corkscrew", "bottle_screw"],
        "synset": "corkscrew.n.01"
    },
    {
        "def": "ears of corn that can be prepared and served for human food",
        "frequency": "c",
        "id": 312,
        "name": "edible_corn",
        "synonyms": ["edible_corn", "corn", "maize"],
        "synset": "corn.n.03"
    },
    {
        "def": "bread made primarily of cornmeal",
        "frequency": "r",
        "id": 313,
        "name": "cornbread",
        "synonyms": ["cornbread"],
        "synset": "cornbread.n.01"
    },
    {
        "def": "a brass musical instrument with a narrow tube and a flared bell and many valves",
        "frequency": "c",
        "id": 314,
        "name": "cornet",
        "synonyms": ["cornet", "horn", "trumpet"],
        "synset": "cornet.n.01"
    },
    {
        "def": "a decorative framework to conceal curtain fixtures at the top of a window casing",
        "frequency": "c",
        "id": 315,
        "name": "cornice",
        "synonyms": ["cornice", "valance", "valance_board", "pelmet"],
        "synset": "cornice.n.01"
    },
    {
        "def": "coarsely ground corn",
        "frequency": "r",
        "id": 316,
        "name": "cornmeal",
        "synonyms": ["cornmeal"],
        "synset": "cornmeal.n.01"
    },
    {
        "def": "a woman's close-fitting foundation garment",
        "frequency": "r",
        "id": 317,
        "name": "corset",
        "synonyms": ["corset", "girdle"],
        "synset": "corset.n.01"
    },
    {
        "def": "lettuce with long dark-green leaves in a loosely packed elongated head",
        "frequency": "r",
        "id": 318,
        "name": "romaine_lettuce",
        "synonyms": ["romaine_lettuce"],
        "synset": "cos.n.02"
    },
    {
        "def": "the attire characteristic of a country or a time or a social class",
        "frequency": "c",
        "id": 319,
        "name": "costume",
        "synonyms": ["costume"],
        "synset": "costume.n.04"
    },
    {
        "def": "large American feline resembling a lion",
        "frequency": "r",
        "id": 320,
        "name": "cougar",
        "synonyms": ["cougar", "puma", "catamount", "mountain_lion", "panther"],
        "synset": "cougar.n.01"
    },
    {
        "def": "a loose-fitting protective garment that is worn over other clothing",
        "frequency": "r",
        "id": 321,
        "name": "coverall",
        "synonyms": ["coverall"],
        "synset": "coverall.n.01"
    },
    {
        "def": "a bell hung around the neck of cow so that the cow can be easily located",
        "frequency": "r",
        "id": 322,
        "name": "cowbell",
        "synonyms": ["cowbell"],
        "synset": "cowbell.n.01"
    },
    {
        "def": "a hat with a wide brim and a soft crown; worn by American ranch hands",
        "frequency": "f",
        "id": 323,
        "name": "cowboy_hat",
        "synonyms": ["cowboy_hat", "ten-gallon_hat"],
        "synset": "cowboy_hat.n.01"
    },
    {
        "def": "decapod having eyes on short stalks and a broad flattened shell and pincers",
        "frequency": "r",
        "id": 324,
        "name": "crab_(animal)",
        "synonyms": ["crab_(animal)"],
        "synset": "crab.n.01"
    },
    {
        "def": "a thin crisp wafer",
        "frequency": "c",
        "id": 325,
        "name": "cracker",
        "synonyms": ["cracker"],
        "synset": "cracker.n.01"
    },
    {
        "def": "small very thin pancake",
        "frequency": "r",
        "id": 326,
        "name": "crape",
        "synonyms": ["crape", "crepe", "French_pancake"],
        "synset": "crape.n.01"
    },
    {
        "def": "a rugged box (usually made of wood); used for shipping",
        "frequency": "f",
        "id": 327,
        "name": "crate",
        "synonyms": ["crate"],
        "synset": "crate.n.01"
    },
    {
        "def": "writing or drawing implement made of a colored stick of composition wax",
        "frequency": "r",
        "id": 328,
        "name": "crayon",
        "synonyms": ["crayon", "wax_crayon"],
        "synset": "crayon.n.01"
    },
    {
        "def": "a small pitcher for serving cream",
        "frequency": "r",
        "id": 329,
        "name": "cream_pitcher",
        "synonyms": ["cream_pitcher"],
        "synset": "cream_pitcher.n.01"
    },
    {
        "def": "a card, usually plastic, used to pay for goods and services",
        "frequency": "r",
        "id": 330,
        "name": "credit_card",
        "synonyms": ["credit_card", "charge_card", "debit_card"],
        "synset": "credit_card.n.01"
    },
    {
        "def": "very rich flaky crescent-shaped roll",
        "frequency": "c",
        "id": 331,
        "name": "crescent_roll",
        "synonyms": ["crescent_roll", "croissant"],
        "synset": "crescent_roll.n.01"
    },
    {
        "def": "baby bed with high sides made of slats",
        "frequency": "c",
        "id": 332,
        "name": "crib",
        "synonyms": ["crib", "cot"],
        "synset": "crib.n.01"
    },
    {
        "def": "an earthen jar (made of baked clay)",
        "frequency": "c",
        "id": 333,
        "name": "crock_pot",
        "synonyms": ["crock_pot", "earthenware_jar"],
        "synset": "crock.n.03"
    },
    {
        "def": "a horizontal bar that goes across something",
        "frequency": "f",
        "id": 334,
        "name": "crossbar",
        "synonyms": ["crossbar"],
        "synset": "crossbar.n.01"
    },
    {
        "def": "a small piece of toasted or fried bread; served in soup or salads",
        "frequency": "r",
        "id": 335,
        "name": "crouton",
        "synonyms": ["crouton"],
        "synset": "crouton.n.01"
    },
    {
        "def": "black birds having a raucous call",
        "frequency": "r",
        "id": 336,
        "name": "crow",
        "synonyms": ["crow"],
        "synset": "crow.n.01"
    },
    {
        "def": "an ornamental jeweled headdress signifying sovereignty",
        "frequency": "c",
        "id": 337,
        "name": "crown",
        "synonyms": ["crown"],
        "synset": "crown.n.04"
    },
    {
        "def": "representation of the cross on which Jesus died",
        "frequency": "c",
        "id": 338,
        "name": "crucifix",
        "synonyms": ["crucifix"],
        "synset": "crucifix.n.01"
    },
    {
        "def": "a passenger ship used commercially for pleasure cruises",
        "frequency": "c",
        "id": 339,
        "name": "cruise_ship",
        "synonyms": ["cruise_ship", "cruise_liner"],
        "synset": "cruise_ship.n.01"
    },
    {
        "def": "a car in which policemen cruise the streets",
        "frequency": "c",
        "id": 340,
        "name": "police_cruiser",
        "synonyms": ["police_cruiser", "patrol_car", "police_car", "squad_car"],
        "synset": "cruiser.n.01"
    },
    {
        "def": "small piece of e.g. bread or cake",
        "frequency": "c",
        "id": 341,
        "name": "crumb",
        "synonyms": ["crumb"],
        "synset": "crumb.n.03"
    },
    {
        "def": "a wooden or metal staff that fits under the armpit and reaches to the ground",
        "frequency": "r",
        "id": 342,
        "name": "crutch",
        "synonyms": ["crutch"],
        "synset": "crutch.n.01"
    },
    {
        "def": "the young of certain carnivorous mammals such as the bear or wolf or lion",
        "frequency": "c",
        "id": 343,
        "name": "cub_(animal)",
        "synonyms": ["cub_(animal)"],
        "synset": "cub.n.03"
    },
    {
        "def": "a block in the (approximate) shape of a cube",
        "frequency": "r",
        "id": 344,
        "name": "cube",
        "synonyms": ["cube", "square_block"],
        "synset": "cube.n.05"
    },
    {
        "def": "cylindrical green fruit with thin green rind and white flesh eaten as a vegetable",
        "frequency": "f",
        "id": 345,
        "name": "cucumber",
        "synonyms": ["cucumber", "cuke"],
        "synset": "cucumber.n.02"
    },
    {
        "def": "jewelry consisting of linked buttons used to fasten the cuffs of a shirt",
        "frequency": "c",
        "id": 346,
        "name": "cufflink",
        "synonyms": ["cufflink"],
        "synset": "cufflink.n.01"
    },
    {
        "def": "a small open container usually used for drinking; usually has a handle",
        "frequency": "f",
        "id": 347,
        "name": "cup",
        "synonyms": ["cup"],
        "synset": "cup.n.01"
    },
    {
        "def": "a metal vessel with handles that is awarded as a trophy to a competition winner",
        "frequency": "c",
        "id": 348,
        "name": "trophy_cup",
        "synonyms": ["trophy_cup"],
        "synset": "cup.n.08"
    },
    {
        "def": "small cake baked in a muffin tin",
        "frequency": "c",
        "id": 349,
        "name": "cupcake",
        "synonyms": ["cupcake"],
        "synset": "cupcake.n.01"
    },
    {
        "def": "a cylindrical tube around which the hair is wound to curl it",
        "frequency": "r",
        "id": 350,
        "name": "hair_curler",
        "synonyms": ["hair_curler", "hair_roller", "hair_crimper"],
        "synset": "curler.n.01"
    },
    {
        "def": "a cylindrical home appliance that heats hair that has been curled around it",
        "frequency": "r",
        "id": 351,
        "name": "curling_iron",
        "synonyms": ["curling_iron"],
        "synset": "curling_iron.n.01"
    },
    {
        "def": "hanging cloth used as a blind (especially for a window)",
        "frequency": "f",
        "id": 352,
        "name": "curtain",
        "synonyms": ["curtain", "drapery"],
        "synset": "curtain.n.01"
    },
    {
        "def": "a soft bag filled with air or padding such as feathers or foam rubber",
        "frequency": "f",
        "id": 353,
        "name": "cushion",
        "synonyms": ["cushion"],
        "synset": "cushion.n.03"
    },
    {
        "def": "sweetened mixture of milk and eggs baked or boiled or frozen",
        "frequency": "r",
        "id": 354,
        "name": "custard",
        "synonyms": ["custard"],
        "synset": "custard.n.01"
    },
    {
        "def": "a cutting implement; a tool for cutting",
        "frequency": "c",
        "id": 355,
        "name": "cutting_tool",
        "synonyms": ["cutting_tool"],
        "synset": "cutter.n.06"
    },
    {
        "def": "a cylindrical container",
        "frequency": "r",
        "id": 356,
        "name": "cylinder",
        "synonyms": ["cylinder"],
        "synset": "cylinder.n.04"
    },
    {
        "def": "a percussion instrument consisting of a concave brass disk",
        "frequency": "r",
        "id": 357,
        "name": "cymbal",
        "synonyms": ["cymbal"],
        "synset": "cymbal.n.01"
    },
    {
        "def": "small long-bodied short-legged breed of dog having a short sleek coat and long "
               "drooping ears",
        "frequency": "r",
        "id": 358,
        "name": "dachshund",
        "synonyms": ["dachshund", "dachsie", "badger_dog"],
        "synset": "dachshund.n.01"
    },
    {
        "def": "a short knife with a pointed blade used for piercing or stabbing",
        "frequency": "r",
        "id": 359,
        "name": "dagger",
        "synonyms": ["dagger"],
        "synset": "dagger.n.01"
    },
    {
        "def": "a circular board of wood or cork used as the target in the game of darts",
        "frequency": "r",
        "id": 360,
        "name": "dartboard",
        "synonyms": ["dartboard"],
        "synset": "dartboard.n.01"
    },
    {
        "def": "sweet edible fruit of the date palm with a single long woody seed",
        "frequency": "r",
        "id": 361,
        "name": "date_(fruit)",
        "synonyms": ["date_(fruit)"],
        "synset": "date.n.08"
    },
    {
        "def": "a folding chair for use outdoors; a wooden frame supports a length of canvas",
        "frequency": "f",
        "id": 362,
        "name": "deck_chair",
        "synonyms": ["deck_chair", "beach_chair"],
        "synset": "deck_chair.n.01"
    },
    {
        "def": "distinguished from Bovidae by the male's having solid deciduous antlers",
        "frequency": "c",
        "id": 363,
        "name": "deer",
        "synonyms": ["deer", "cervid"],
        "synset": "deer.n.01"
    },
    {
        "def": "a soft thread for cleaning the spaces between the teeth",
        "frequency": "c",
        "id": 364,
        "name": "dental_floss",
        "synonyms": ["dental_floss", "floss"],
        "synset": "dental_floss.n.01"
    },
    {
        "def": "a piece of furniture with a writing surface and usually drawers or other "
               "compartments",
        "frequency": "f",
        "id": 365,
        "name": "desk",
        "synonyms": ["desk"],
        "synset": "desk.n.01"
    },
    {
        "def": "a surface-active chemical widely used in industry and laundering",
        "frequency": "r",
        "id": 366,
        "name": "detergent",
        "synonyms": ["detergent"],
        "synset": "detergent.n.01"
    },
    {
        "def": "garment consisting of a folded cloth drawn up between the legs and fastened at the "
               "waist",
        "frequency": "c",
        "id": 367,
        "name": "diaper",
        "synonyms": ["diaper"],
        "synset": "diaper.n.01"
    },
    {
        "def": "a daily written record of (usually personal) experiences and observations",
        "frequency": "r",
        "id": 368,
        "name": "diary",
        "synonyms": ["diary", "journal"],
        "synset": "diary.n.01"
    },
    {
        "def": "a small cube with 1 to 6 spots on the six faces; used in gambling",
        "frequency": "r",
        "id": 369,
        "name": "die",
        "synonyms": ["die", "dice"],
        "synset": "die.n.01"
    },
    {
        "def": "a small boat of shallow draft with seats and oars with which it is propelled",
        "frequency": "r",
        "id": 370,
        "name": "dinghy",
        "synonyms": ["dinghy", "dory", "rowboat"],
        "synset": "dinghy.n.01"
    },
    {
        "def": "a table at which meals are served",
        "frequency": "f",
        "id": 371,
        "name": "dining_table",
        "synonyms": ["dining_table"],
        "synset": "dining_table.n.01"
    },
    {
        "def": "semiformal evening dress for men",
        "frequency": "r",
        "id": 372,
        "name": "tux",
        "synonyms": ["tux", "tuxedo"],
        "synset": "dinner_jacket.n.01"
    },
    {
        "def": "a piece of dishware normally used as a container for holding or serving food",
        "frequency": "c",
        "id": 373,
        "name": "dish",
        "synonyms": ["dish"],
        "synset": "dish.n.01"
    },
    {
        "def": "directional antenna consisting of a parabolic reflector",
        "frequency": "c",
        "id": 374,
        "name": "dish_antenna",
        "synonyms": ["dish_antenna"],
        "synset": "dish.n.05"
    },
    {
        "def": "a cloth for washing dishes",
        "frequency": "c",
        "id": 375,
        "name": "dishrag",
        "synonyms": ["dishrag", "dishcloth"],
        "synset": "dishrag.n.01"
    },
    {
        "def": "a towel for drying dishes",
        "frequency": "c",
        "id": 376,
        "name": "dishtowel",
        "synonyms": ["dishtowel", "tea_towel"],
        "synset": "dishtowel.n.01"
    },
    {
        "def": "a machine for washing dishes",
        "frequency": "f",
        "id": 377,
        "name": "dishwasher",
        "synonyms": ["dishwasher", "dishwashing_machine"],
        "synset": "dishwasher.n.01"
    },
    {
        "def": "a low-sudsing detergent designed for use in dishwashers",
        "frequency": "r",
        "id": 378,
        "name": "dishwasher_detergent",
        "synonyms": ["dishwasher_detergent", "dishwashing_detergent", "dishwashing_liquid"],
        "synset": "dishwasher_detergent.n.01"
    },
    {
        "def": "a small plastic magnetic disk enclosed in a stiff envelope used to store data",
        "frequency": "r",
        "id": 379,
        "name": "diskette",
        "synonyms": ["diskette", "floppy", "floppy_disk"],
        "synset": "diskette.n.01"
    },
    {
        "def": "a container so designed that the contents can be used in prescribed amounts",
        "frequency": "c",
        "id": 380,
        "name": "dispenser",
        "synonyms": ["dispenser"],
        "synset": "dispenser.n.01"
    },
    {
        "def": "a disposable cup made of paper; for holding drinks",
        "frequency": "c",
        "id": 381,
        "name": "Dixie_cup",
        "synonyms": ["Dixie_cup", "paper_cup"],
        "synset": "dixie_cup.n.01"
    },
    {
        "def": "a common domesticated dog",
        "frequency": "f",
        "id": 382,
        "name": "dog",
        "synonyms": ["dog"],
        "synset": "dog.n.01"
    },
    {
        "def": "a collar for a dog",
        "frequency": "f",
        "id": 383,
        "name": "dog_collar",
        "synonyms": ["dog_collar"],
        "synset": "dog_collar.n.01"
    },
    {
        "def": "a toy replica of a HUMAN (NOT AN ANIMAL)",
        "frequency": "c",
        "id": 384,
        "name": "doll",
        "synonyms": ["doll"],
        "synset": "doll.n.01"
    },
    {
        "def": "a piece of paper money worth one dollar",
        "frequency": "r",
        "id": 385,
        "name": "dollar",
        "synonyms": ["dollar", "dollar_bill", "one_dollar_bill"],
        "synset": "dollar.n.02"
    },
    {
        "def": "any of various small toothed whales with a beaklike snout; larger than porpoises",
        "frequency": "r",
        "id": 386,
        "name": "dolphin",
        "synonyms": ["dolphin"],
        "synset": "dolphin.n.02"
    },
    {
        "def": "domestic beast of burden descended from the African wild ass; patient but stubborn",
        "frequency": "c",
        "id": 387,
        "name": "domestic_ass",
        "synonyms": ["domestic_ass", "donkey"],
        "synset": "domestic_ass.n.01"
    },
    {
        "def": "a mask covering the upper part of the face but with holes for the eyes",
        "frequency": "r",
        "id": 388,
        "name": "eye_mask",
        "synonyms": ["eye_mask"],
        "synset": "domino.n.03"
    },
    {
        "def": "a button at an outer door that gives a ringing or buzzing signal when pushed",
        "frequency": "r",
        "id": 389,
        "name": "doorbell",
        "synonyms": ["doorbell", "buzzer"],
        "synset": "doorbell.n.01"
    },
    {
        "def": "a knob used to open a door (often called `doorhandle' in Great Britain)",
        "frequency": "f",
        "id": 390,
        "name": "doorknob",
        "synonyms": ["doorknob", "doorhandle"],
        "synset": "doorknob.n.01"
    },
    {
        "def": "a mat placed outside an exterior door for wiping the shoes before entering",
        "frequency": "c",
        "id": 391,
        "name": "doormat",
        "synonyms": ["doormat", "welcome_mat"],
        "synset": "doormat.n.02"
    },
    {
        "def": "a small ring-shaped friedcake",
        "frequency": "f",
        "id": 392,
        "name": "doughnut",
        "synonyms": ["doughnut", "donut"],
        "synset": "doughnut.n.02"
    },
    {
        "def": "any of numerous small pigeons",
        "frequency": "r",
        "id": 393,
        "name": "dove",
        "synonyms": ["dove"],
        "synset": "dove.n.01"
    },
    {
        "def": "slender-bodied non-stinging insect having iridescent wings that are outspread at "
               "rest",
        "frequency": "r",
        "id": 394,
        "name": "dragonfly",
        "synonyms": ["dragonfly"],
        "synset": "dragonfly.n.01"
    },
    {
        "def": "a boxlike container in a piece of furniture; made so as to slide in and out",
        "frequency": "f",
        "id": 395,
        "name": "drawer",
        "synonyms": ["drawer"],
        "synset": "drawer.n.01"
    },
    {
        "def": "underpants worn by men",
        "frequency": "c",
        "id": 396,
        "name": "underdrawers",
        "synonyms": ["underdrawers", "boxers", "boxershorts"],
        "synset": "drawers.n.01"
    },
    {
        "def": "a one-piece garment for a woman; has skirt and bodice",
        "frequency": "f",
        "id": 397,
        "name": "dress",
        "synonyms": ["dress", "frock"],
        "synset": "dress.n.01"
    },
    {
        "def": "a man's hat with a tall crown; usually covered with silk or with beaver fur",
        "frequency": "c",
        "id": 398,
        "name": "dress_hat",
        "synonyms": ["dress_hat", "high_hat", "opera_hat", "silk_hat", "top_hat"],
        "synset": "dress_hat.n.01"
    },
    {
        "def": "formalwear consisting of full evening dress for men",
        "frequency": "c",
        "id": 399,
        "name": "dress_suit",
        "synonyms": ["dress_suit"],
        "synset": "dress_suit.n.01"
    },
    {
        "def": "a cabinet with shelves",
        "frequency": "c",
        "id": 400,
        "name": "dresser",
        "synonyms": ["dresser"],
        "synset": "dresser.n.05"
    },
    {
        "def": "a tool with a sharp rotating point for making holes in hard materials",
        "frequency": "c",
        "id": 401,
        "name": "drill",
        "synonyms": ["drill"],
        "synset": "drill.n.01"
    },
    {
        "def": "a public fountain to provide a jet of drinking water",
        "frequency": "r",
        "id": 402,
        "name": "drinking_fountain",
        "synonyms": ["drinking_fountain"],
        "synset": "drinking_fountain.n.01"
    },
    {
        "def": "an aircraft without a pilot that is operated by remote control",
        "frequency": "r",
        "id": 403,
        "name": "drone",
        "synonyms": ["drone"],
        "synset": "drone.n.04"
    },
    {
        "def": "pipet consisting of a small tube with a vacuum bulb at one end for drawing liquid "
               "in and releasing it a drop at a time",
        "frequency": "r",
        "id": 404,
        "name": "dropper",
        "synonyms": ["dropper", "eye_dropper"],
        "synset": "dropper.n.01"
    },
    {
        "def": "a musical percussion instrument; usually consists of a hollow cylinder with a "
               "membrane stretched across each end",
        "frequency": "c",
        "id": 405,
        "name": "drum_(musical_instrument)",
        "synonyms": ["drum_(musical_instrument)"],
        "synset": "drum.n.01"
    },
    {
        "def": "a stick used for playing a drum",
        "frequency": "r",
        "id": 406,
        "name": "drumstick",
        "synonyms": ["drumstick"],
        "synset": "drumstick.n.02"
    },
    {
        "def": "small web-footed broad-billed swimming bird",
        "frequency": "f",
        "id": 407,
        "name": "duck",
        "synonyms": ["duck"],
        "synset": "duck.n.01"
    },
    {
        "def": "young duck",
        "frequency": "r",
        "id": 408,
        "name": "duckling",
        "synonyms": ["duckling"],
        "synset": "duckling.n.02"
    },
    {
        "def": "a wide silvery adhesive tape",
        "frequency": "c",
        "id": 409,
        "name": "duct_tape",
        "synonyms": ["duct_tape"],
        "synset": "duct_tape.n.01"
    },
    {
        "def": "a large cylindrical bag of heavy cloth",
        "frequency": "f",
        "id": 410,
        "name": "duffel_bag",
        "synonyms": ["duffel_bag", "duffle_bag", "duffel", "duffle"],
        "synset": "duffel_bag.n.01"
    },
    {
        "def": "an exercising weight with two ball-like ends connected by a short handle",
        "frequency": "r",
        "id": 411,
        "name": "dumbbell",
        "synonyms": ["dumbbell"],
        "synset": "dumbbell.n.01"
    },
    {
        "def": "a container designed to receive and transport and dump waste",
        "frequency": "c",
        "id": 412,
        "name": "dumpster",
        "synonyms": ["dumpster"],
        "synset": "dumpster.n.01"
    },
    {
        "def": "a short-handled receptacle into which dust can be swept",
        "frequency": "r",
        "id": 413,
        "name": "dustpan",
        "synonyms": ["dustpan"],
        "synset": "dustpan.n.02"
    },
    {
        "def": "iron or earthenware cooking pot; used for stews",
        "frequency": "r",
        "id": 414,
        "name": "Dutch_oven",
        "synonyms": ["Dutch_oven"],
        "synset": "dutch_oven.n.02"
    },
    {
        "def": "large birds of prey noted for their broad wings and strong soaring flight",
        "frequency": "c",
        "id": 415,
        "name": "eagle",
        "synonyms": ["eagle"],
        "synset": "eagle.n.01"
    },
    {
        "def": "device for listening to audio that is held over or inserted into the ear",
        "frequency": "f",
        "id": 416,
        "name": "earphone",
        "synonyms": ["earphone", "earpiece", "headphone"],
        "synset": "earphone.n.01"
    },
    {
        "def": "a soft plug that is inserted into the ear canal to block sound",
        "frequency": "r",
        "id": 417,
        "name": "earplug",
        "synonyms": ["earplug"],
        "synset": "earplug.n.01"
    },
    {
        "def": "jewelry to ornament the ear",
        "frequency": "f",
        "id": 418,
        "name": "earring",
        "synonyms": ["earring"],
        "synset": "earring.n.01"
    },
    {
        "def": "an upright tripod for displaying something (usually an artist's canvas)",
        "frequency": "c",
        "id": 419,
        "name": "easel",
        "synonyms": ["easel"],
        "synset": "easel.n.01"
    },
    {
        "def": "oblong cream puff",
        "frequency": "r",
        "id": 420,
        "name": "eclair",
        "synonyms": ["eclair"],
        "synset": "eclair.n.01"
    },
    {
        "def": "an elongate fish with fatty flesh",
        "frequency": "r",
        "id": 421,
        "name": "eel",
        "synonyms": ["eel"],
        "synset": "eel.n.01"
    },
    {
        "def": "oval reproductive body of a fowl (especially a hen) used as food",
        "frequency": "f",
        "id": 422,
        "name": "egg",
        "synonyms": ["egg", "eggs"],
        "synset": "egg.n.02"
    },
    {
        "def": "minced vegetables and meat wrapped in a pancake and fried",
        "frequency": "r",
        "id": 423,
        "name": "egg_roll",
        "synonyms": ["egg_roll", "spring_roll"],
        "synset": "egg_roll.n.01"
    },
    {
        "def": "the yellow spherical part of an egg",
        "frequency": "c",
        "id": 424,
        "name": "egg_yolk",
        "synonyms": ["egg_yolk", "yolk_(egg)"],
        "synset": "egg_yolk.n.01"
    },
    {
        "def": "a mixer for beating eggs or whipping cream",
        "frequency": "c",
        "id": 425,
        "name": "eggbeater",
        "synonyms": ["eggbeater", "eggwhisk"],
        "synset": "eggbeater.n.02"
    },
    {
        "def": "egg-shaped vegetable having a shiny skin typically dark purple",
        "frequency": "c",
        "id": 426,
        "name": "eggplant",
        "synonyms": ["eggplant", "aubergine"],
        "synset": "eggplant.n.01"
    },
    {
        "def": "a chair-shaped instrument of execution by electrocution",
        "frequency": "r",
        "id": 427,
        "name": "electric_chair",
        "synonyms": ["electric_chair"],
        "synset": "electric_chair.n.01"
    },
    {
        "def": "a refrigerator in which the coolant is pumped around by an electric motor",
        "frequency": "f",
        "id": 428,
        "name": "refrigerator",
        "synonyms": ["refrigerator"],
        "synset": "electric_refrigerator.n.01"
    },
    {
        "def": "a common elephant",
        "frequency": "f",
        "id": 429,
        "name": "elephant",
        "synonyms": ["elephant"],
        "synset": "elephant.n.01"
    },
    {
        "def": "large northern deer with enormous flattened antlers in the male",
        "frequency": "r",
        "id": 430,
        "name": "elk",
        "synonyms": ["elk", "moose"],
        "synset": "elk.n.01"
    },
    {
        "def": "a flat (usually rectangular) container for a letter, thin package, etc.",
        "frequency": "c",
        "id": 431,
        "name": "envelope",
        "synonyms": ["envelope"],
        "synset": "envelope.n.01"
    },
    {
        "def": "an implement used to erase something",
        "frequency": "c",
        "id": 432,
        "name": "eraser",
        "synonyms": ["eraser"],
        "synset": "eraser.n.01"
    },
    {
        "def": "edible snail usually served in the shell with a sauce of melted butter and garlic",
        "frequency": "r",
        "id": 433,
        "name": "escargot",
        "synonyms": ["escargot"],
        "synset": "escargot.n.01"
    },
    {
        "def": "a protective cloth covering for an injured eye",
        "frequency": "r",
        "id": 434,
        "name": "eyepatch",
        "synonyms": ["eyepatch"],
        "synset": "eyepatch.n.01"
    },
    {
        "def": "birds of prey having long pointed powerful wings adapted for swift flight",
        "frequency": "r",
        "id": 435,
        "name": "falcon",
        "synonyms": ["falcon"],
        "synset": "falcon.n.01"
    },
    {
        "def": "a device for creating a current of air by movement of a surface or surfaces",
        "frequency": "f",
        "id": 436,
        "name": "fan",
        "synonyms": ["fan"],
        "synset": "fan.n.01"
    },
    {
        "def": "a regulator for controlling the flow of a liquid from a reservoir",
        "frequency": "f",
        "id": 437,
        "name": "faucet",
        "synonyms": ["faucet", "spigot", "tap"],
        "synset": "faucet.n.01"
    },
    {
        "def": "a hat made of felt with a creased crown",
        "frequency": "r",
        "id": 438,
        "name": "fedora",
        "synonyms": ["fedora"],
        "synset": "fedora.n.01"
    },
    {
        "def": "domesticated albino variety of the European polecat bred for hunting rats and "
               "rabbits",
        "frequency": "r",
        "id": 439,
        "name": "ferret",
        "synonyms": ["ferret"],
        "synset": "ferret.n.02"
    },
    {
        "def": "a large wheel with suspended seats that remain upright as the wheel rotates",
        "frequency": "c",
        "id": 440,
        "name": "Ferris_wheel",
        "synonyms": ["Ferris_wheel"],
        "synset": "ferris_wheel.n.01"
    },
    {
        "def": "a boat that transports people or vehicles across a body of water and operates on a "
               "regular schedule",
        "frequency": "r",
        "id": 441,
        "name": "ferry",
        "synonyms": ["ferry", "ferryboat"],
        "synset": "ferry.n.01"
    },
    {
        "def": "fleshy sweet pear-shaped yellowish or purple fruit eaten fresh or preserved or "
               "dried",
        "frequency": "r",
        "id": 442,
        "name": "fig_(fruit)",
        "synonyms": ["fig_(fruit)"],
        "synset": "fig.n.04"
    },
    {
        "def": "a high-speed military or naval airplane designed to destroy enemy targets",
        "frequency": "c",
        "id": 443,
        "name": "fighter_jet",
        "synonyms": ["fighter_jet", "fighter_aircraft", "attack_aircraft"],
        "synset": "fighter.n.02"
    },
    {
        "def": "a small carved or molded figure",
        "frequency": "f",
        "id": 444,
        "name": "figurine",
        "synonyms": ["figurine"],
        "synset": "figurine.n.01"
    },
    {
        "def": "office furniture consisting of a container for keeping papers in order",
        "frequency": "c",
        "id": 445,
        "name": "file_cabinet",
        "synonyms": ["file_cabinet", "filing_cabinet"],
        "synset": "file.n.03"
    },
    {
        "def": "a steel hand tool with small sharp teeth on some or all of its surfaces; used for "
               "smoothing wood or metal",
        "frequency": "r",
        "id": 446,
        "name": "file_(tool)",
        "synonyms": ["file_(tool)"],
        "synset": "file.n.04"
    },
    {
        "def": "an alarm that is tripped off by fire or smoke",
        "frequency": "f",
        "id": 447,
        "name": "fire_alarm",
        "synonyms": ["fire_alarm", "smoke_alarm"],
        "synset": "fire_alarm.n.02"
    },
    {
        "def": "large trucks that carry firefighters and equipment to the site of a fire",
        "frequency": "c",
        "id": 448,
        "name": "fire_engine",
        "synonyms": ["fire_engine", "fire_truck"],
        "synset": "fire_engine.n.01"
    },
    {
        "def": "a manually operated device for extinguishing small fires",
        "frequency": "c",
        "id": 449,
        "name": "fire_extinguisher",
        "synonyms": ["fire_extinguisher", "extinguisher"],
        "synset": "fire_extinguisher.n.01"
    },
    {
        "def": "a large hose that carries water from a fire hydrant to the site of the fire",
        "frequency": "c",
        "id": 450,
        "name": "fire_hose",
        "synonyms": ["fire_hose"],
        "synset": "fire_hose.n.01"
    },
    {
        "def": "an open recess in a wall at the base of a chimney where a fire can be built",
        "frequency": "f",
        "id": 451,
        "name": "fireplace",
        "synonyms": ["fireplace"],
        "synset": "fireplace.n.01"
    },
    {
        "def": "an upright hydrant for drawing water to use in fighting a fire",
        "frequency": "f",
        "id": 452,
        "name": "fireplug",
        "synonyms": ["fireplug", "fire_hydrant", "hydrant"],
        "synset": "fireplug.n.01"
    },
    {
        "def": "any of various mostly cold-blooded aquatic vertebrates usually having scales and "
               "breathing through gills",
        "frequency": "c",
        "id": 453,
        "name": "fish",
        "synonyms": ["fish"],
        "synset": "fish.n.01"
    },
    {
        "def": "the flesh of fish used as food",
        "frequency": "r",
        "id": 454,
        "name": "fish_(food)",
        "synonyms": ["fish_(food)"],
        "synset": "fish.n.02"
    },
    {
        "def": "a transparent bowl in which small fish are kept",
        "frequency": "r",
        "id": 455,
        "name": "fishbowl",
        "synonyms": ["fishbowl", "goldfish_bowl"],
        "synset": "fishbowl.n.02"
    },
    {
        "def": "a vessel for fishing",
        "frequency": "r",
        "id": 456,
        "name": "fishing_boat",
        "synonyms": ["fishing_boat", "fishing_vessel"],
        "synset": "fishing_boat.n.01"
    },
    {
        "def": "a rod that is used in fishing to extend the fishing line",
        "frequency": "c",
        "id": 457,
        "name": "fishing_rod",
        "synonyms": ["fishing_rod", "fishing_pole"],
        "synset": "fishing_rod.n.01"
    },
    {
        "def": "emblem usually consisting of a rectangular piece of cloth of distinctive design "
               "(do not include pole)",
        "frequency": "f",
        "id": 458,
        "name": "flag",
        "synonyms": ["flag"],
        "synset": "flag.n.01"
    },
    {
        "def": "a tall staff or pole on which a flag is raised",
        "frequency": "f",
        "id": 459,
        "name": "flagpole",
        "synonyms": ["flagpole", "flagstaff"],
        "synset": "flagpole.n.02"
    },
    {
        "def": "large pink web-footed bird with down-bent bill",
        "frequency": "c",
        "id": 460,
        "name": "flamingo",
        "synonyms": ["flamingo"],
        "synset": "flamingo.n.01"
    },
    {
        "def": "a soft light woolen fabric; used for clothing",
        "frequency": "c",
        "id": 461,
        "name": "flannel",
        "synonyms": ["flannel"],
        "synset": "flannel.n.01"
    },
    {
        "def": "a lamp for providing momentary light to take a photograph",
        "frequency": "r",
        "id": 462,
        "name": "flash",
        "synonyms": ["flash", "flashbulb"],
        "synset": "flash.n.10"
    },
    {
        "def": "a small portable battery-powered electric lamp",
        "frequency": "c",
        "id": 463,
        "name": "flashlight",
        "synonyms": ["flashlight", "torch"],
        "synset": "flashlight.n.01"
    },
    {
        "def": "a soft bulky fabric with deep pile; used chiefly for clothing",
        "frequency": "r",
        "id": 464,
        "name": "fleece",
        "synonyms": ["fleece"],
        "synset": "fleece.n.03"
    },
    {
        "def": "a backless sandal held to the foot by a thong between two toes",
        "frequency": "f",
        "id": 465,
        "name": "flip-flop_(sandal)",
        "synonyms": ["flip-flop_(sandal)"],
        "synset": "flip-flop.n.02"
    },
    {
        "def": "a shoe to aid a person in swimming",
        "frequency": "c",
        "id": 466,
        "name": "flipper_(footwear)",
        "synonyms": ["flipper_(footwear)", "fin_(footwear)"],
        "synset": "flipper.n.01"
    },
    {
        "def": "a decorative arrangement of flowers",
        "frequency": "f",
        "id": 467,
        "name": "flower_arrangement",
        "synonyms": ["flower_arrangement", "floral_arrangement"],
        "synset": "flower_arrangement.n.01"
    },
    {
        "def": "a tall narrow wineglass",
        "frequency": "c",
        "id": 468,
        "name": "flute_glass",
        "synonyms": ["flute_glass", "champagne_flute"],
        "synset": "flute.n.02"
    },
    {
        "def": "a young horse",
        "frequency": "r",
        "id": 469,
        "name": "foal",
        "synonyms": ["foal"],
        "synset": "foal.n.01"
    },
    {
        "def": "a chair that can be folded flat for storage",
        "frequency": "c",
        "id": 470,
        "name": "folding_chair",
        "synonyms": ["folding_chair"],
        "synset": "folding_chair.n.01"
    },
    {
        "def": "a kitchen appliance for shredding, blending, chopping, or slicing food",
        "frequency": "c",
        "id": 471,
        "name": "food_processor",
        "synonyms": ["food_processor"],
        "synset": "food_processor.n.01"
    },
    {
        "def": "the inflated oblong ball used in playing American football",
        "frequency": "c",
        "id": 472,
        "name": "football_(American)",
        "synonyms": ["football_(American)"],
        "synset": "football.n.02"
    },
    {
        "def": "a padded helmet with a face mask to protect the head of football players",
        "frequency": "r",
        "id": 473,
        "name": "football_helmet",
        "synonyms": ["football_helmet"],
        "synset": "football_helmet.n.01"
    },
    {
        "def": "a low seat or a stool to rest the feet of a seated person",
        "frequency": "c",
        "id": 474,
        "name": "footstool",
        "synonyms": ["footstool", "footrest"],
        "synset": "footstool.n.01"
    },
    {
        "def": "cutlery used for serving and eating food",
        "frequency": "f",
        "id": 475,
        "name": "fork",
        "synonyms": ["fork"],
        "synset": "fork.n.01"
    },
    {
        "def": "an industrial vehicle with a power operated fork in front that can be inserted "
               "under loads to lift and move them",
        "frequency": "r",
        "id": 476,
        "name": "forklift",
        "synonyms": ["forklift"],
        "synset": "forklift.n.01"
    },
    {
        "def": "a railway car that carries freight",
        "frequency": "r",
        "id": 477,
        "name": "freight_car",
        "synonyms": ["freight_car"],
        "synset": "freight_car.n.01"
    },
    {
        "def": "bread slice dipped in egg and milk and fried",
        "frequency": "r",
        "id": 478,
        "name": "French_toast",
        "synonyms": ["French_toast"],
        "synset": "french_toast.n.01"
    },
    {
        "def": "anything that freshens",
        "frequency": "c",
        "id": 479,
        "name": "freshener",
        "synonyms": ["freshener", "air_freshener"],
        "synset": "freshener.n.01"
    },
    {
        "def": "a light, plastic disk propelled with a flip of the wrist for recreation or "
               "competition",
        "frequency": "f",
        "id": 480,
        "name": "frisbee",
        "synonyms": ["frisbee"],
        "synset": "frisbee.n.01"
    },
    {
        "def": "a tailless stout-bodied amphibians with long hind limbs for leaping",
        "frequency": "c",
        "id": 481,
        "name": "frog",
        "synonyms": ["frog", "toad", "toad_frog"],
        "synset": "frog.n.01"
    },
    {
        "def": "drink produced by squeezing or crushing fruit",
        "frequency": "c",
        "id": 482,
        "name": "fruit_juice",
        "synonyms": ["fruit_juice"],
        "synset": "fruit_juice.n.01"
    },
    {
        "def": "salad composed of fruits",
        "frequency": "r",
        "id": 483,
        "name": "fruit_salad",
        "synonyms": ["fruit_salad"],
        "synset": "fruit_salad.n.01"
    },
    {
        "def": "a pan used for frying foods",
        "frequency": "c",
        "id": 484,
        "name": "frying_pan",
        "synonyms": ["frying_pan", "frypan", "skillet"],
        "synset": "frying_pan.n.01"
    },
    {
        "def": "soft creamy candy",
        "frequency": "r",
        "id": 485,
        "name": "fudge",
        "synonyms": ["fudge"],
        "synset": "fudge.n.01"
    },
    {
        "def": "a cone-shaped utensil used to channel a substance into a container with a small "
               "mouth",
        "frequency": "r",
        "id": 486,
        "name": "funnel",
        "synonyms": ["funnel"],
        "synset": "funnel.n.02"
    },
    {
        "def": "a pad that is used for sleeping on the floor or on a raised frame",
        "frequency": "c",
        "id": 487,
        "name": "futon",
        "synonyms": ["futon"],
        "synset": "futon.n.01"
    },
    {
        "def": "restraint put into a person's mouth to prevent speaking or shouting",
        "frequency": "r",
        "id": 488,
        "name": "gag",
        "synonyms": ["gag", "muzzle"],
        "synset": "gag.n.02"
    },
    {
        "def": "a receptacle where waste can be discarded",
        "frequency": "r",
        "id": 489,
        "name": "garbage",
        "synonyms": ["garbage"],
        "synset": "garbage.n.03"
    },
    {
        "def": "a truck for collecting domestic refuse",
        "frequency": "c",
        "id": 490,
        "name": "garbage_truck",
        "synonyms": ["garbage_truck"],
        "synset": "garbage_truck.n.01"
    },
    {
        "def": "a hose used for watering a lawn or garden",
        "frequency": "c",
        "id": 491,
        "name": "garden_hose",
        "synonyms": ["garden_hose"],
        "synset": "garden_hose.n.01"
    },
    {
        "def": "a medicated solution used for gargling and rinsing the mouth",
        "frequency": "c",
        "id": 492,
        "name": "gargle",
        "synonyms": ["gargle", "mouthwash"],
        "synset": "gargle.n.01"
    },
    {
        "def": "an ornament consisting of a grotesquely carved figure of a person or animal",
        "frequency": "r",
        "id": 493,
        "name": "gargoyle",
        "synonyms": ["gargoyle"],
        "synset": "gargoyle.n.02"
    },
    {
        "def": "aromatic bulb used as seasoning",
        "frequency": "c",
        "id": 494,
        "name": "garlic",
        "synonyms": ["garlic", "ail"],
        "synset": "garlic.n.02"
    },
    {
        "def": "a protective face mask with a filter",
        "frequency": "r",
        "id": 495,
        "name": "gasmask",
        "synonyms": ["gasmask", "respirator", "gas_helmet"],
        "synset": "gasmask.n.01"
    },
    {
        "def": "small swift graceful antelope of Africa and Asia having lustrous eyes",
        "frequency": "r",
        "id": 496,
        "name": "gazelle",
        "synonyms": ["gazelle"],
        "synset": "gazelle.n.01"
    },
    {
        "def": "an edible jelly made with gelatin and used as a dessert or salad base or a coating "
               "for foods",
        "frequency": "c",
        "id": 497,
        "name": "gelatin",
        "synonyms": ["gelatin", "jelly"],
        "synset": "gelatin.n.02"
    },
    {
        "def": "a crystalline rock that can be cut and polished for jewelry",
        "frequency": "r",
        "id": 498,
        "name": "gemstone",
        "synonyms": ["gemstone"],
        "synset": "gem.n.02"
    },
    {
        "def": "large black-and-white herbivorous mammal of bamboo forests of China and Tibet",
        "frequency": "c",
        "id": 499,
        "name": "giant_panda",
        "synonyms": ["giant_panda", "panda", "panda_bear"],
        "synset": "giant_panda.n.01"
    },
    {
        "def": "attractive wrapping paper suitable for wrapping gifts",
        "frequency": "c",
        "id": 500,
        "name": "gift_wrap",
        "synonyms": ["gift_wrap"],
        "synset": "gift_wrap.n.01"
    },
    {
        "def": "the root of the common ginger plant; used fresh as a seasoning",
        "frequency": "c",
        "id": 501,
        "name": "ginger",
        "synonyms": ["ginger", "gingerroot"],
        "synset": "ginger.n.03"
    },
    {
        "def": "tall animal having a spotted coat and small horns and very long neck and legs",
        "frequency": "f",
        "id": 502,
        "name": "giraffe",
        "synonyms": ["giraffe"],
        "synset": "giraffe.n.01"
    },
    {
        "def": "a band of material around the waist that strengthens a skirt or trousers",
        "frequency": "c",
        "id": 503,
        "name": "cincture",
        "synonyms": ["cincture", "sash", "waistband", "waistcloth"],
        "synset": "girdle.n.02"
    },
    {
        "def": "a container for holding liquids while drinking",
        "frequency": "f",
        "id": 504,
        "name": "glass_(drink_container)",
        "synonyms": ["glass_(drink_container)", "drinking_glass"],
        "synset": "glass.n.02"
    },
    {
        "def": "a sphere on which a map (especially of the earth) is represented",
        "frequency": "c",
        "id": 505,
        "name": "globe",
        "synonyms": ["globe"],
        "synset": "globe.n.03"
    },
    {
        "def": "handwear covering the hand",
        "frequency": "f",
        "id": 506,
        "name": "glove",
        "synonyms": ["glove"],
        "synset": "glove.n.02"
    },
    {
        "def": "a common goat",
        "frequency": "c",
        "id": 507,
        "name": "goat",
        "synonyms": ["goat"],
        "synset": "goat.n.01"
    },
    {
        "def": "tight-fitting spectacles worn to protect the eyes",
        "frequency": "f",
        "id": 508,
        "name": "goggles",
        "synonyms": ["goggles"],
        "synset": "goggles.n.01"
    },
    {
        "def": "small golden or orange-red freshwater fishes used as pond or aquarium pets",
        "frequency": "r",
        "id": 509,
        "name": "goldfish",
        "synonyms": ["goldfish"],
        "synset": "goldfish.n.01"
    },
    {
        "def": "golf equipment used by a golfer to hit a golf ball",
        "frequency": "r",
        "id": 510,
        "name": "golf_club",
        "synonyms": ["golf_club", "golf-club"],
        "synset": "golf_club.n.02"
    },
    {
        "def": "a small motor vehicle in which golfers can ride between shots",
        "frequency": "c",
        "id": 511,
        "name": "golfcart",
        "synonyms": ["golfcart"],
        "synset": "golfcart.n.01"
    },
    {
        "def": "long narrow flat-bottomed boat propelled by sculling; traditionally used on canals "
               "of Venice",
        "frequency": "r",
        "id": 512,
        "name": "gondola_(boat)",
        "synonyms": ["gondola_(boat)"],
        "synset": "gondola.n.02"
    },
    {
        "def": "loud, web-footed long-necked aquatic birds usually larger than ducks",
        "frequency": "c",
        "id": 513,
        "name": "goose",
        "synonyms": ["goose"],
        "synset": "goose.n.01"
    },
    {
        "def": "largest ape",
        "frequency": "r",
        "id": 514,
        "name": "gorilla",
        "synonyms": ["gorilla"],
        "synset": "gorilla.n.01"
    },
    {
        "def": "any of numerous inedible fruits with hard rinds",
        "frequency": "r",
        "id": 515,
        "name": "gourd",
        "synonyms": ["gourd"],
        "synset": "gourd.n.02"
    },
    {
        "def": "protective garment worn by surgeons during operations",
        "frequency": "r",
        "id": 516,
        "name": "surgical_gown",
        "synonyms": ["surgical_gown", "scrubs_(surgical_clothing)"],
        "synset": "gown.n.04"
    },
    {
        "def": "any of various juicy fruit with green or purple skins; grow in clusters",
        "frequency": "f",
        "id": 517,
        "name": "grape",
        "synonyms": ["grape"],
        "synset": "grape.n.01"
    },
    {
        "def": "plant-eating insect with hind legs adapted for leaping",
        "frequency": "r",
        "id": 518,
        "name": "grasshopper",
        "synonyms": ["grasshopper"],
        "synset": "grasshopper.n.01"
    },
    {
        "def": "utensil with sharp perforations for shredding foods (as vegetables or cheese)",
        "frequency": "c",
        "id": 519,
        "name": "grater",
        "synonyms": ["grater"],
        "synset": "grater.n.01"
    },
    {
        "def": "a stone that is used to mark a grave",
        "frequency": "c",
        "id": 520,
        "name": "gravestone",
        "synonyms": ["gravestone", "headstone", "tombstone"],
        "synset": "gravestone.n.01"
    },
    {
        "def": "a dish (often boat-shaped) for serving gravy or sauce",
        "frequency": "r",
        "id": 521,
        "name": "gravy_boat",
        "synonyms": ["gravy_boat", "gravy_holder"],
        "synset": "gravy_boat.n.01"
    },
    {
        "def": "a common bean plant cultivated for its slender green edible pods",
        "frequency": "c",
        "id": 522,
        "name": "green_bean",
        "synonyms": ["green_bean"],
        "synset": "green_bean.n.02"
    },
    {
        "def": "a young onion before the bulb has enlarged",
        "frequency": "c",
        "id": 523,
        "name": "green_onion",
        "synonyms": ["green_onion", "spring_onion", "scallion"],
        "synset": "green_onion.n.01"
    },
    {
        "def": "cooking utensil consisting of a flat heated surface on which food is cooked",
        "frequency": "r",
        "id": 524,
        "name": "griddle",
        "synonyms": ["griddle"],
        "synset": "griddle.n.01"
    },
    {
        "def": "a restaurant where food is cooked on a grill",
        "frequency": "r",
        "id": 525,
        "name": "grillroom",
        "synonyms": ["grillroom", "grill_(restaurant)"],
        "synset": "grillroom.n.01"
    },
    {
        "def": "a machine tool that polishes metal",
        "frequency": "r",
        "id": 526,
        "name": "grinder_(tool)",
        "synonyms": ["grinder_(tool)"],
        "synset": "grinder.n.04"
    },
    {
        "def": "coarsely ground corn boiled as a breakfast dish",
        "frequency": "r",
        "id": 527,
        "name": "grits",
        "synonyms": ["grits", "hominy_grits"],
        "synset": "grits.n.01"
    },
    {
        "def": "powerful brownish-yellow bear of the uplands of western North America",
        "frequency": "c",
        "id": 528,
        "name": "grizzly",
        "synonyms": ["grizzly", "grizzly_bear"],
        "synset": "grizzly.n.01"
    },
    {
        "def": "a sack for holding customer's groceries",
        "frequency": "c",
        "id": 529,
        "name": "grocery_bag",
        "synonyms": ["grocery_bag"],
        "synset": "grocery_bag.n.01"
    },
    {
        "def": "a dip made of mashed avocado mixed with chopped onions and other seasonings",
        "frequency": "r",
        "id": 530,
        "name": "guacamole",
        "synonyms": ["guacamole"],
        "synset": "guacamole.n.01"
    },
    {
        "def": "a stringed instrument usually having six strings; played by strumming or plucking",
        "frequency": "f",
        "id": 531,
        "name": "guitar",
        "synonyms": ["guitar"],
        "synset": "guitar.n.01"
    },
    {
        "def": "mostly white aquatic bird having long pointed wings and short legs",
        "frequency": "c",
        "id": 532,
        "name": "gull",
        "synonyms": ["gull", "seagull"],
        "synset": "gull.n.02"
    },
    {
        "def": "a weapon that discharges a bullet at high velocity from a metal tube",
        "frequency": "c",
        "id": 533,
        "name": "gun",
        "synonyms": ["gun"],
        "synset": "gun.n.01"
    },
    {
        "def": "substance sprayed on the hair to hold it in place",
        "frequency": "r",
        "id": 534,
        "name": "hair_spray",
        "synonyms": ["hair_spray"],
        "synset": "hair_spray.n.01"
    },
    {
        "def": "a brush used to groom a person's hair",
        "frequency": "c",
        "id": 535,
        "name": "hairbrush",
        "synonyms": ["hairbrush"],
        "synset": "hairbrush.n.01"
    },
    {
        "def": "a small net that someone wears over their hair to keep it in place",
        "frequency": "c",
        "id": 536,
        "name": "hairnet",
        "synonyms": ["hairnet"],
        "synset": "hairnet.n.01"
    },
    {
        "def": "a double pronged pin used to hold women's hair in place",
        "frequency": "c",
        "id": 537,
        "name": "hairpin",
        "synonyms": ["hairpin"],
        "synset": "hairpin.n.01"
    },
    {
        "def": "meat cut from the thigh of a hog (usually smoked)",
        "frequency": "f",
        "id": 538,
        "name": "ham",
        "synonyms": ["ham", "jambon", "gammon"],
        "synset": "ham.n.01"
    },
    {
        "def": "a sandwich consisting of a patty of minced beef served on a bun",
        "frequency": "c",
        "id": 539,
        "name": "hamburger",
        "synonyms": ["hamburger", "beefburger", "burger"],
        "synset": "hamburger.n.01"
    },
    {
        "def": "a hand tool with a heavy head and a handle; used to deliver an impulsive force by "
               "striking",
        "frequency": "c",
        "id": 540,
        "name": "hammer",
        "synonyms": ["hammer"],
        "synset": "hammer.n.02"
    },
    {
        "def": "a hanging bed of canvas or rope netting (usually suspended between two trees)",
        "frequency": "r",
        "id": 541,
        "name": "hammock",
        "synonyms": ["hammock"],
        "synset": "hammock.n.02"
    },
    {
        "def": "a basket usually with a cover",
        "frequency": "r",
        "id": 542,
        "name": "hamper",
        "synonyms": ["hamper"],
        "synset": "hamper.n.02"
    },
    {
        "def": "short-tailed burrowing rodent with large cheek pouches",
        "frequency": "r",
        "id": 543,
        "name": "hamster",
        "synonyms": ["hamster"],
        "synset": "hamster.n.01"
    },
    {
        "def": "a hand-held electric blower that can blow warm air onto the hair",
        "frequency": "c",
        "id": 544,
        "name": "hair_dryer",
        "synonyms": ["hair_dryer"],
        "synset": "hand_blower.n.01"
    },
    {
        "def": "a mirror intended to be held in the hand",
        "frequency": "r",
        "id": 545,
        "name": "hand_glass",
        "synonyms": ["hand_glass", "hand_mirror"],
        "synset": "hand_glass.n.01"
    },
    {
        "def": "a small towel used to dry the hands or face",
        "frequency": "f",
        "id": 546,
        "name": "hand_towel",
        "synonyms": ["hand_towel", "face_towel"],
        "synset": "hand_towel.n.01"
    },
    {
        "def": "wheeled vehicle that can be pushed by a person",
        "frequency": "c",
        "id": 547,
        "name": "handcart",
        "synonyms": ["handcart", "pushcart", "hand_truck"],
        "synset": "handcart.n.01"
    },
    {
        "def": "shackle that consists of a metal loop that can be locked around the wrist",
        "frequency": "r",
        "id": 548,
        "name": "handcuff",
        "synonyms": ["handcuff"],
        "synset": "handcuff.n.01"
    },
    {
        "def": "a square piece of cloth used for wiping the eyes or nose or as a costume accessory",
        "frequency": "c",
        "id": 549,
        "name": "handkerchief",
        "synonyms": ["handkerchief"],
        "synset": "handkerchief.n.01"
    },
    {
        "def": "the appendage to an object that is designed to be held in order to use or move it",
        "frequency": "f",
        "id": 550,
        "name": "handle",
        "synonyms": ["handle", "grip", "handgrip"],
        "synset": "handle.n.01"
    },
    {
        "def": "a saw used with one hand for cutting wood",
        "frequency": "r",
        "id": 551,
        "name": "handsaw",
        "synonyms": ["handsaw", "carpenter's_saw"],
        "synset": "handsaw.n.01"
    },
    {
        "def": "a book with cardboard or cloth or leather covers",
        "frequency": "r",
        "id": 552,
        "name": "hardback_book",
        "synonyms": ["hardback_book", "hardcover_book"],
        "synset": "hardback.n.01"
    },
    {
        "def": "a free-reed instrument in which air is forced through the reeds by bellows",
        "frequency": "r",
        "id": 553,
        "name": "harmonium",
        "synonyms": ["harmonium", "organ_(musical_instrument)", "reed_organ_(musical_instrument)"],
        "synset": "harmonium.n.01"
    },
    {
        "def": "headwear that protects the head from bad weather, sun, or worn for fashion",
        "frequency": "f",
        "id": 554,
        "name": "hat",
        "synonyms": ["hat"],
        "synset": "hat.n.01"
    },
    {
        "def": "a round piece of luggage for carrying hats",
        "frequency": "r",
        "id": 555,
        "name": "hatbox",
        "synonyms": ["hatbox"],
        "synset": "hatbox.n.01"
    },
    {
        "def": "a movable barrier covering a hatchway",
        "frequency": "r",
        "id": 556,
        "name": "hatch",
        "synonyms": ["hatch"],
        "synset": "hatch.n.03"
    },
    {
        "def": "a garment that covers the head and face",
        "frequency": "c",
        "id": 557,
        "name": "veil",
        "synonyms": ["veil"],
        "synset": "head_covering.n.01"
    },
    {
        "def": "a band worn around or over the head",
        "frequency": "f",
        "id": 558,
        "name": "headband",
        "synonyms": ["headband"],
        "synset": "headband.n.01"
    },
    {
        "def": "a vertical board or panel forming the head of a bedstead",
        "frequency": "f",
        "id": 559,
        "name": "headboard",
        "synonyms": ["headboard"],
        "synset": "headboard.n.01"
    },
    {
        "def": "a powerful light with reflector; attached to the front of an automobile or "
               "locomotive",
        "frequency": "f",
        "id": 560,
        "name": "headlight",
        "synonyms": ["headlight", "headlamp"],
        "synset": "headlight.n.01"
    },
    {
        "def": "a kerchief worn over the head and tied under the chin",
        "frequency": "c",
        "id": 561,
        "name": "headscarf",
        "synonyms": ["headscarf"],
        "synset": "headscarf.n.01"
    },
    {
        "def": "receiver consisting of a pair of headphones",
        "frequency": "r",
        "id": 562,
        "name": "headset",
        "synonyms": ["headset"],
        "synset": "headset.n.01"
    },
    {
        "def": "the band that is the part of a bridle that fits around a horse's head",
        "frequency": "c",
        "id": 563,
        "name": "headstall_(for_horses)",
        "synonyms": ["headstall_(for_horses)", "headpiece_(for_horses)"],
        "synset": "headstall.n.01"
    },
    {
        "def": "an acoustic device used to direct sound to the ear of a hearing-impaired person",
        "frequency": "r",
        "id": 564,
        "name": "hearing_aid",
        "synonyms": ["hearing_aid"],
        "synset": "hearing_aid.n.02"
    },
    {
        "def": "a muscular organ; its contractions move the blood through the body",
        "frequency": "c",
        "id": 565,
        "name": "heart",
        "synonyms": ["heart"],
        "synset": "heart.n.02"
    },
    {
        "def": "device that heats water or supplies warmth to a room",
        "frequency": "c",
        "id": 566,
        "name": "heater",
        "synonyms": ["heater", "warmer"],
        "synset": "heater.n.01"
    },
    {
        "def": "an aircraft without wings that obtains its lift from the rotation of overhead "
               "blades",
        "frequency": "c",
        "id": 567,
        "name": "helicopter",
        "synonyms": ["helicopter"],
        "synset": "helicopter.n.01"
    },
    {
        "def": "a protective headgear made of hard material to resist blows",
        "frequency": "f",
        "id": 568,
        "name": "helmet",
        "synonyms": ["helmet"],
        "synset": "helmet.n.02"
    },
    {
        "def": "grey or white wading bird with long neck and long legs and (usually) long bill",
        "frequency": "r",
        "id": 569,
        "name": "heron",
        "synonyms": ["heron"],
        "synset": "heron.n.02"
    },
    {
        "def": "a chair for feeding a very young child",
        "frequency": "c",
        "id": 570,
        "name": "highchair",
        "synonyms": ["highchair", "feeding_chair"],
        "synset": "highchair.n.01"
    },
    {
        "def": "a joint that holds two parts together so that one can swing relative to the other",
        "frequency": "f",
        "id": 571,
        "name": "hinge",
        "synonyms": ["hinge"],
        "synset": "hinge.n.01"
    },
    {
        "def": "massive thick-skinned animal living in or around rivers of tropical Africa",
        "frequency": "r",
        "id": 572,
        "name": "hippopotamus",
        "synonyms": ["hippopotamus"],
        "synset": "hippopotamus.n.01"
    },
    {
        "def": "sports implement consisting of a stick used by hockey players to move the puck",
        "frequency": "r",
        "id": 573,
        "name": "hockey_stick",
        "synonyms": ["hockey_stick"],
        "synset": "hockey_stick.n.01"
    },
    {
        "def": "domestic swine",
        "frequency": "c",
        "id": 574,
        "name": "hog",
        "synonyms": ["hog", "pig"],
        "synset": "hog.n.03"
    },
    {
        "def": "(baseball) a rubber slab where the batter stands; it must be touched by a base "
               "runner in order to score",
        "frequency": "f",
        "id": 575,
        "name": "home_plate_(baseball)",
        "synonyms": ["home_plate_(baseball)", "home_base_(baseball)"],
        "synset": "home_plate.n.01"
    },
    {
        "def": "a sweet yellow liquid produced by bees",
        "frequency": "c",
        "id": 576,
        "name": "honey",
        "synonyms": ["honey"],
        "synset": "honey.n.01"
    },
    {
        "def": "metal covering leading to a vent that exhausts smoke or fumes",
        "frequency": "f",
        "id": 577,
        "name": "fume_hood",
        "synonyms": ["fume_hood", "exhaust_hood"],
        "synset": "hood.n.06"
    },
    {
        "def": "a curved or bent implement for suspending or pulling something",
        "frequency": "f",
        "id": 578,
        "name": "hook",
        "synonyms": ["hook"],
        "synset": "hook.n.05"
    },
    {
        "def": "a common horse",
        "frequency": "f",
        "id": 579,
        "name": "horse",
        "synonyms": ["horse"],
        "synset": "horse.n.01"
    },
    {
        "def": "a flexible pipe for conveying a liquid or gas",
        "frequency": "f",
        "id": 580,
        "name": "hose",
        "synonyms": ["hose", "hosepipe"],
        "synset": "hose.n.03"
    },
    {
        "def": "balloon for travel through the air in a basket suspended below a large bag of "
               "heated air",
        "frequency": "r",
        "id": 581,
        "name": "hot-air_balloon",
        "synonyms": ["hot-air_balloon"],
        "synset": "hot-air_balloon.n.01"
    },
    {
        "def": "a portable electric appliance for heating or cooking or keeping food warm",
        "frequency": "r",
        "id": 582,
        "name": "hotplate",
        "synonyms": ["hotplate"],
        "synset": "hot_plate.n.01"
    },
    {
        "def": "a pungent peppery sauce",
        "frequency": "c",
        "id": 583,
        "name": "hot_sauce",
        "synonyms": ["hot_sauce"],
        "synset": "hot_sauce.n.01"
    },
    {
        "def": "a sandglass timer that runs for sixty minutes",
        "frequency": "r",
        "id": 584,
        "name": "hourglass",
        "synonyms": ["hourglass"],
        "synset": "hourglass.n.01"
    },
    {
        "def": "a barge that is designed and equipped for use as a dwelling",
        "frequency": "r",
        "id": 585,
        "name": "houseboat",
        "synonyms": ["houseboat"],
        "synset": "houseboat.n.01"
    },
    {
        "def": "tiny American bird having brilliant iridescent plumage and long slender bills",
        "frequency": "r",
        "id": 586,
        "name": "hummingbird",
        "synonyms": ["hummingbird"],
        "synset": "hummingbird.n.01"
    },
    {
        "def": "a thick spread made from mashed chickpeas",
        "frequency": "r",
        "id": 587,
        "name": "hummus",
        "synonyms": ["hummus", "humus", "hommos", "hoummos", "humous"],
        "synset": "hummus.n.01"
    },
    {
        "def": "white bear of Arctic regions",
        "frequency": "c",
        "id": 588,
        "name": "polar_bear",
        "synonyms": ["polar_bear"],
        "synset": "ice_bear.n.01"
    },
    {
        "def": "frozen dessert containing cream and sugar and flavoring",
        "frequency": "c",
        "id": 589,
        "name": "icecream",
        "synonyms": ["icecream"],
        "synset": "ice_cream.n.01"
    },
    {
        "def": "ice cream or water ice on a small wooden stick",
        "frequency": "r",
        "id": 590,
        "name": "popsicle",
        "synonyms": ["popsicle"],
        "synset": "ice_lolly.n.01"
    },
    {
        "def": "an appliance included in some electric refrigerators for making ice cubes",
        "frequency": "c",
        "id": 591,
        "name": "ice_maker",
        "synonyms": ["ice_maker"],
        "synset": "ice_maker.n.01"
    },
    {
        "def": "a waterproof bag filled with ice: applied to the body (especially the head) to "
               "cool or reduce swelling",
        "frequency": "r",
        "id": 592,
        "name": "ice_pack",
        "synonyms": ["ice_pack", "ice_bag"],
        "synset": "ice_pack.n.01"
    },
    {
        "def": "skate consisting of a boot with a steel blade fitted to the sole",
        "frequency": "r",
        "id": 593,
        "name": "ice_skate",
        "synonyms": ["ice_skate"],
        "synset": "ice_skate.n.01"
    },
    {
        "def": "strong tea served over ice",
        "frequency": "r",
        "id": 594,
        "name": "ice_tea",
        "synonyms": ["ice_tea", "iced_tea"],
        "synset": "ice_tea.n.01"
    },
    {
        "def": "a substance or device used to start a fire",
        "frequency": "c",
        "id": 595,
        "name": "igniter",
        "synonyms": ["igniter", "ignitor", "lighter"],
        "synset": "igniter.n.01"
    },
    {
        "def": "a substance that produces a fragrant odor when burned",
        "frequency": "r",
        "id": 596,
        "name": "incense",
        "synonyms": ["incense"],
        "synset": "incense.n.01"
    },
    {
        "def": "a dispenser that produces a chemical vapor to be inhaled through mouth or nose",
        "frequency": "r",
        "id": 597,
        "name": "inhaler",
        "synonyms": ["inhaler", "inhalator"],
        "synset": "inhaler.n.01"
    },
    {
        "def": "a pocket-sized device used to play music files",
        "frequency": "c",
        "id": 598,
        "name": "iPod",
        "synonyms": ["iPod"],
        "synset": "ipod.n.01"
    },
    {
        "def": "home appliance consisting of a flat metal base that is heated and used to smooth "
               "cloth",
        "frequency": "c",
        "id": 599,
        "name": "iron_(for_clothing)",
        "synonyms": ["iron_(for_clothing)", "smoothing_iron_(for_clothing)"],
        "synset": "iron.n.04"
    },
    {
        "def": "narrow padded board on collapsible supports; used for ironing clothes",
        "frequency": "r",
        "id": 600,
        "name": "ironing_board",
        "synonyms": ["ironing_board"],
        "synset": "ironing_board.n.01"
    },
    {
        "def": "a waist-length coat",
        "frequency": "f",
        "id": 601,
        "name": "jacket",
        "synonyms": ["jacket"],
        "synset": "jacket.n.01"
    },
    {
        "def": "preserve of crushed fruit",
        "frequency": "r",
        "id": 602,
        "name": "jam",
        "synonyms": ["jam"],
        "synset": "jam.n.01"
    },
    {
        "def": "(usually plural) close-fitting trousers of heavy denim for manual work or casual "
               "wear",
        "frequency": "f",
        "id": 603,
        "name": "jean",
        "synonyms": ["jean", "blue_jean", "denim"],
        "synset": "jean.n.01"
    },
    {
        "def": "a car suitable for traveling over rough terrain",
        "frequency": "c",
        "id": 604,
        "name": "jeep",
        "synonyms": ["jeep", "landrover"],
        "synset": "jeep.n.01"
    },
    {
        "def": "sugar-glazed jellied candy",
        "frequency": "r",
        "id": 605,
        "name": "jelly_bean",
        "synonyms": ["jelly_bean", "jelly_egg"],
        "synset": "jelly_bean.n.01"
    },
    {
        "def": "a close-fitting pullover shirt",
        "frequency": "f",
        "id": 606,
        "name": "jersey",
        "synonyms": ["jersey", "T-shirt", "tee_shirt"],
        "synset": "jersey.n.03"
    },
    {
        "def": "an airplane powered by one or more jet engines",
        "frequency": "c",
        "id": 607,
        "name": "jet_plane",
        "synonyms": ["jet_plane", "jet-propelled_plane"],
        "synset": "jet.n.01"
    },
    {
        "def": "an adornment (as a bracelet or ring or necklace) made of precious metals and set "
               "with gems (or imitation gems)",
        "frequency": "c",
        "id": 608,
        "name": "jewelry",
        "synonyms": ["jewelry", "jewellery"],
        "synset": "jewelry.n.01"
    },
    {
        "def": "a control device for computers consisting of a vertical handle that can move "
               "freely in two directions",
        "frequency": "r",
        "id": 609,
        "name": "joystick",
        "synonyms": ["joystick"],
        "synset": "joystick.n.02"
    },
    {
        "def": "one-piece garment fashioned after a parachutist's uniform",
        "frequency": "r",
        "id": 610,
        "name": "jumpsuit",
        "synonyms": ["jumpsuit"],
        "synset": "jump_suit.n.01"
    },
    {
        "def": "a small canoe consisting of a light frame made watertight with animal skins",
        "frequency": "c",
        "id": 611,
        "name": "kayak",
        "synonyms": ["kayak"],
        "synset": "kayak.n.01"
    },
    {
        "def": "small cask or barrel",
        "frequency": "r",
        "id": 612,
        "name": "keg",
        "synonyms": ["keg"],
        "synset": "keg.n.02"
    },
    {
        "def": "outbuilding that serves as a shelter for a dog",
        "frequency": "r",
        "id": 613,
        "name": "kennel",
        "synonyms": ["kennel", "doghouse"],
        "synset": "kennel.n.01"
    },
    {
        "def": "a metal pot for stewing or boiling; usually has a lid",
        "frequency": "c",
        "id": 614,
        "name": "kettle",
        "synonyms": ["kettle", "boiler"],
        "synset": "kettle.n.01"
    },
    {
        "def": "metal instrument used to unlock a lock",
        "frequency": "f",
        "id": 615,
        "name": "key",
        "synonyms": ["key"],
        "synset": "key.n.01"
    },
    {
        "def": "a plastic card used to gain access typically to a door",
        "frequency": "r",
        "id": 616,
        "name": "keycard",
        "synonyms": ["keycard"],
        "synset": "keycard.n.01"
    },
    {
        "def": "a knee-length pleated tartan skirt worn by men as part of the traditional dress in "
               "the Highlands of northern Scotland",
        "frequency": "r",
        "id": 617,
        "name": "kilt",
        "synonyms": ["kilt"],
        "synset": "kilt.n.01"
    },
    {
        "def": "a loose robe; imitated from robes originally worn by Japanese",
        "frequency": "c",
        "id": 618,
        "name": "kimono",
        "synonyms": ["kimono"],
        "synset": "kimono.n.01"
    },
    {
        "def": "a sink in a kitchen",
        "frequency": "f",
        "id": 619,
        "name": "kitchen_sink",
        "synonyms": ["kitchen_sink"],
        "synset": "kitchen_sink.n.01"
    },
    {
        "def": "a table in the kitchen",
        "frequency": "c",
        "id": 620,
        "name": "kitchen_table",
        "synonyms": ["kitchen_table"],
        "synset": "kitchen_table.n.01"
    },
    {
        "def": "plaything consisting of a light frame covered with tissue paper; flown in wind at "
               "end of a string",
        "frequency": "f",
        "id": 621,
        "name": "kite",
        "synonyms": ["kite"],
        "synset": "kite.n.03"
    },
    {
        "def": "young domestic cat",
        "frequency": "c",
        "id": 622,
        "name": "kitten",
        "synonyms": ["kitten", "kitty"],
        "synset": "kitten.n.01"
    },
    {
        "def": "fuzzy brown egg-shaped fruit with slightly tart green flesh",
        "frequency": "c",
        "id": 623,
        "name": "kiwi_fruit",
        "synonyms": ["kiwi_fruit"],
        "synset": "kiwi.n.03"
    },
    {
        "def": "protective garment consisting of a pad worn by football or baseball or hockey "
               "players",
        "frequency": "f",
        "id": 624,
        "name": "knee_pad",
        "synonyms": ["knee_pad"],
        "synset": "knee_pad.n.01"
    },
    {
        "def": "tool with a blade and point used as a cutting instrument",
        "frequency": "f",
        "id": 625,
        "name": "knife",
        "synonyms": ["knife"],
        "synset": "knife.n.01"
    },
    {
        "def": "a chess game piece shaped to resemble the head of a horse",
        "frequency": "r",
        "id": 626,
        "name": "knight_(chess_piece)",
        "synonyms": ["knight_(chess_piece)", "horse_(chess_piece)"],
        "synset": "knight.n.02"
    },
    {
        "def": "needle consisting of a slender rod with pointed ends; usually used in pairs",
        "frequency": "r",
        "id": 627,
        "name": "knitting_needle",
        "synonyms": ["knitting_needle"],
        "synset": "knitting_needle.n.01"
    },
    {
        "def": "a round handle often found on a door",
        "frequency": "f",
        "id": 628,
        "name": "knob",
        "synonyms": ["knob"],
        "synset": "knob.n.02"
    },
    {
        "def": "a device (usually metal and ornamental) attached by a hinge to a door",
        "frequency": "r",
        "id": 629,
        "name": "knocker_(on_a_door)",
        "synonyms": ["knocker_(on_a_door)", "doorknocker"],
        "synset": "knocker.n.05"
    },
    {
        "def": "sluggish tailless Australian marsupial with grey furry ears and coat",
        "frequency": "r",
        "id": 630,
        "name": "koala",
        "synonyms": ["koala", "koala_bear"],
        "synset": "koala.n.01"
    },
    {
        "def": "a light coat worn to protect clothing from substances used while working in a "
               "laboratory",
        "frequency": "r",
        "id": 631,
        "name": "lab_coat",
        "synonyms": ["lab_coat", "laboratory_coat"],
        "synset": "lab_coat.n.01"
    },
    {
        "def": "steps consisting of two parallel members connected by rungs",
        "frequency": "f",
        "id": 632,
        "name": "ladder",
        "synonyms": ["ladder"],
        "synset": "ladder.n.01"
    },
    {
        "def": "a spoon-shaped vessel with a long handle frequently used to transfer liquids",
        "frequency": "c",
        "id": 633,
        "name": "ladle",
        "synonyms": ["ladle"],
        "synset": "ladle.n.01"
    },
    {
        "def": "small round bright-colored and spotted beetle, typically red and black",
        "frequency": "r",
        "id": 634,
        "name": "ladybug",
        "synonyms": ["ladybug", "ladybeetle", "ladybird_beetle"],
        "synset": "ladybug.n.01"
    },
    {
        "def": "young sheep",
        "frequency": "c",
        "id": 635,
        "name": "lamb_(animal)",
        "synonyms": ["lamb_(animal)"],
        "synset": "lamb.n.01"
    },
    {
        "def": "chop cut from a lamb",
        "frequency": "r",
        "id": 636,
        "name": "lamb-chop",
        "synonyms": ["lamb-chop", "lambchop"],
        "synset": "lamb_chop.n.01"
    },
    {
        "def": "a piece of furniture holding one or more electric light bulbs",
        "frequency": "f",
        "id": 637,
        "name": "lamp",
        "synonyms": ["lamp"],
        "synset": "lamp.n.02"
    },
    {
        "def": "a metal post supporting an outdoor lamp (such as a streetlight)",
        "frequency": "f",
        "id": 638,
        "name": "lamppost",
        "synonyms": ["lamppost"],
        "synset": "lamppost.n.01"
    },
    {
        "def": "a protective ornamental shade used to screen a light bulb from direct view",
        "frequency": "f",
        "id": 639,
        "name": "lampshade",
        "synonyms": ["lampshade"],
        "synset": "lampshade.n.01"
    },
    {
        "def": "light in a transparent protective case",
        "frequency": "c",
        "id": 640,
        "name": "lantern",
        "synonyms": ["lantern"],
        "synset": "lantern.n.01"
    },
    {
        "def": "a cord worn around the neck to hold a knife or whistle, etc.",
        "frequency": "f",
        "id": 641,
        "name": "lanyard",
        "synonyms": ["lanyard", "laniard"],
        "synset": "lanyard.n.02"
    },
    {
        "def": "a portable computer small enough to use in your lap",
        "frequency": "f",
        "id": 642,
        "name": "laptop_computer",
        "synonyms": ["laptop_computer", "notebook_computer"],
        "synset": "laptop.n.01"
    },
    {
        "def": "baked dish of layers of lasagna pasta with sauce and cheese and meat or vegetables",
        "frequency": "r",
        "id": 643,
        "name": "lasagna",
        "synonyms": ["lasagna", "lasagne"],
        "synset": "lasagna.n.01"
    },
    {
        "def": "a bar that can be lowered or slid into a groove to fasten a door or gate",
        "frequency": "c",
        "id": 644,
        "name": "latch",
        "synonyms": ["latch"],
        "synset": "latch.n.02"
    },
    {
        "def": "garden tool for mowing grass on lawns",
        "frequency": "r",
        "id": 645,
        "name": "lawn_mower",
        "synonyms": ["lawn_mower"],
        "synset": "lawn_mower.n.01"
    },
    {
        "def": "an animal skin made smooth and flexible by removing the hair and then tanning",
        "frequency": "r",
        "id": 646,
        "name": "leather",
        "synonyms": ["leather"],
        "synset": "leather.n.01"
    },
    {
        "def": "a garment covering the leg (usually extending from the knee to the ankle)",
        "frequency": "c",
        "id": 647,
        "name": "legging_(clothing)",
        "synonyms": ["legging_(clothing)", "leging_(clothing)", "leg_covering"],
        "synset": "legging.n.01"
    },
    {
        "def": "a child's plastic construction set for making models from blocks",
        "frequency": "c",
        "id": 648,
        "name": "Lego",
        "synonyms": ["Lego", "Lego_set"],
        "synset": "lego.n.01"
    },
    {
        "def": "yellow oval fruit with juicy acidic flesh",
        "frequency": "f",
        "id": 649,
        "name": "lemon",
        "synonyms": ["lemon"],
        "synset": "lemon.n.01"
    },
    {
        "def": "sweetened beverage of diluted lemon juice",
        "frequency": "r",
        "id": 650,
        "name": "lemonade",
        "synonyms": ["lemonade"],
        "synset": "lemonade.n.01"
    },
    {
        "def": "leafy plant commonly eaten in salad or on sandwiches",
        "frequency": "f",
        "id": 651,
        "name": "lettuce",
        "synonyms": ["lettuce"],
        "synset": "lettuce.n.02"
    },
    {
        "def": "a plate mounted on the front and back of car and bearing the car's registration "
               "number",
        "frequency": "f",
        "id": 652,
        "name": "license_plate",
        "synonyms": ["license_plate", "numberplate"],
        "synset": "license_plate.n.01"
    },
    {
        "def": "a ring-shaped life preserver used to prevent drowning (NOT a life-jacket or vest)",
        "frequency": "f",
        "id": 653,
        "name": "life_buoy",
        "synonyms": ["life_buoy", "lifesaver", "life_belt", "life_ring"],
        "synset": "life_buoy.n.01"
    },
    {
        "def": "life preserver consisting of a sleeveless jacket of buoyant or inflatable design",
        "frequency": "f",
        "id": 654,
        "name": "life_jacket",
        "synonyms": ["life_jacket", "life_vest"],
        "synset": "life_jacket.n.01"
    },
    {
        "def": "glass bulb or tube shaped electric device that emits light (DO NOT MARK LAMPS AS A "
               "WHOLE)",
        "frequency": "f",
        "id": 655,
        "name": "lightbulb",
        "synonyms": ["lightbulb"],
        "synset": "light_bulb.n.01"
    },
    {
        "def": "a metallic conductor that is attached to a high point and leads to the ground",
        "frequency": "r",
        "id": 656,
        "name": "lightning_rod",
        "synonyms": ["lightning_rod", "lightning_conductor"],
        "synset": "lightning_rod.n.02"
    },
    {
        "def": "the green acidic fruit of any of various lime trees",
        "frequency": "c",
        "id": 657,
        "name": "lime",
        "synonyms": ["lime"],
        "synset": "lime.n.06"
    },
    {
        "def": "long luxurious car; usually driven by a chauffeur",
        "frequency": "r",
        "id": 658,
        "name": "limousine",
        "synonyms": ["limousine"],
        "synset": "limousine.n.01"
    },
    {
        "def": "a high-quality paper made of linen fibers or with a linen finish",
        "frequency": "r",
        "id": 659,
        "name": "linen_paper",
        "synonyms": ["linen_paper"],
        "synset": "linen.n.02"
    },
    {
        "def": "large gregarious predatory cat of Africa and India",
        "frequency": "c",
        "id": 660,
        "name": "lion",
        "synonyms": ["lion"],
        "synset": "lion.n.01"
    },
    {
        "def": "a balm applied to the lips",
        "frequency": "c",
        "id": 661,
        "name": "lip_balm",
        "synonyms": ["lip_balm"],
        "synset": "lip_balm.n.01"
    },
    {
        "def": "makeup that is used to color the lips",
        "frequency": "c",
        "id": 662,
        "name": "lipstick",
        "synonyms": ["lipstick", "lip_rouge"],
        "synset": "lipstick.n.01"
    },
    {
        "def": "an alcoholic beverage that is distilled rather than fermented",
        "frequency": "r",
        "id": 663,
        "name": "liquor",
        "synonyms": ["liquor", "spirits", "hard_liquor", "liqueur", "cordial"],
        "synset": "liquor.n.01"
    },
    {
        "def": "a reptile with usually two pairs of legs and a tapering tail",
        "frequency": "r",
        "id": 664,
        "name": "lizard",
        "synonyms": ["lizard"],
        "synset": "lizard.n.01"
    },
    {
        "def": "a low leather step-in shoe",
        "frequency": "r",
        "id": 665,
        "name": "Loafer_(type_of_shoe)",
        "synonyms": ["Loafer_(type_of_shoe)"],
        "synset": "loafer.n.02"
    },
    {
        "def": "a segment of the trunk of a tree when stripped of branches",
        "frequency": "f",
        "id": 666,
        "name": "log",
        "synonyms": ["log"],
        "synset": "log.n.01"
    },
    {
        "def": "hard candy on a stick",
        "frequency": "c",
        "id": 667,
        "name": "lollipop",
        "synonyms": ["lollipop"],
        "synset": "lollipop.n.02"
    },
    {
        "def": "any of various cosmetic preparations that are applied to the skin",
        "frequency": "c",
        "id": 668,
        "name": "lotion",
        "synonyms": ["lotion"],
        "synset": "lotion.n.01"
    },
    {
        "def": "electronic device that produces sound often as part of a stereo system",
        "frequency": "f",
        "id": 669,
        "name": "speaker_(stero_equipment)",
        "synonyms": ["speaker_(stero_equipment)"],
        "synset": "loudspeaker.n.01"
    },
    {
        "def": "small sofa that seats two people",
        "frequency": "c",
        "id": 670,
        "name": "loveseat",
        "synonyms": ["loveseat"],
        "synset": "love_seat.n.01"
    },
    {
        "def": "a rapidly firing automatic gun",
        "frequency": "r",
        "id": 671,
        "name": "machine_gun",
        "synonyms": ["machine_gun"],
        "synset": "machine_gun.n.01"
    },
    {
        "def": "a paperback periodic publication",
        "frequency": "f",
        "id": 672,
        "name": "magazine",
        "synonyms": ["magazine"],
        "synset": "magazine.n.02"
    },
    {
        "def": "a device that attracts iron and produces a magnetic field",
        "frequency": "f",
        "id": 673,
        "name": "magnet",
        "synonyms": ["magnet"],
        "synset": "magnet.n.01"
    },
    {
        "def": "a slot (usually in a door) through which mail can be delivered",
        "frequency": "r",
        "id": 674,
        "name": "mail_slot",
        "synonyms": ["mail_slot"],
        "synset": "mail_slot.n.01"
    },
    {
        "def": "a private box for delivery of mail",
        "frequency": "c",
        "id": 675,
        "name": "mailbox_(at_home)",
        "synonyms": ["mailbox_(at_home)", "letter_box_(at_home)"],
        "synset": "mailbox.n.01"
    },
    {
        "def": "a sports implement with a long handle and a hammer-like head used to hit a ball",
        "frequency": "r",
        "id": 676,
        "name": "mallet",
        "synonyms": ["mallet"],
        "synset": "mallet.n.01"
    },
    {
        "def": "any of numerous extinct elephants widely distributed in the Pleistocene",
        "frequency": "r",
        "id": 677,
        "name": "mammoth",
        "synonyms": ["mammoth"],
        "synset": "mammoth.n.01"
    },
    {
        "def": "a somewhat flat reddish-orange loose skinned citrus of China",
        "frequency": "c",
        "id": 678,
        "name": "mandarin_orange",
        "synonyms": ["mandarin_orange"],
        "synset": "mandarin.n.05"
    },
    {
        "def": "a container (usually in a barn or stable) from which cattle or horses feed",
        "frequency": "c",
        "id": 679,
        "name": "manger",
        "synonyms": ["manger", "trough"],
        "synset": "manger.n.01"
    },
    {
        "def": "a hole (usually with a flush cover) through which a person can gain access to an "
               "underground structure",
        "frequency": "f",
        "id": 680,
        "name": "manhole",
        "synonyms": ["manhole"],
        "synset": "manhole.n.01"
    },
    {
        "def": "a diagrammatic representation of the earth's surface (or part of it)",
        "frequency": "c",
        "id": 681,
        "name": "map",
        "synonyms": ["map"],
        "synset": "map.n.01"
    },
    {
        "def": "a writing implement for making a mark",
        "frequency": "c",
        "id": 682,
        "name": "marker",
        "synonyms": ["marker"],
        "synset": "marker.n.03"
    },
    {
        "def": "a cocktail made of gin (or vodka) with dry vermouth",
        "frequency": "r",
        "id": 683,
        "name": "martini",
        "synonyms": ["martini"],
        "synset": "martini.n.01"
    },
    {
        "def": "a person or animal that is adopted by a team or other group as a symbolic figure",
        "frequency": "r",
        "id": 684,
        "name": "mascot",
        "synonyms": ["mascot"],
        "synset": "mascot.n.01"
    },
    {
        "def": "potato that has been peeled and boiled and then mashed",
        "frequency": "c",
        "id": 685,
        "name": "mashed_potato",
        "synonyms": ["mashed_potato"],
        "synset": "mashed_potato.n.01"
    },
    {
        "def": "a kitchen utensil used for mashing (e.g. potatoes)",
        "frequency": "r",
        "id": 686,
        "name": "masher",
        "synonyms": ["masher"],
        "synset": "masher.n.02"
    },
    {
        "def": "a protective covering worn over the face",
        "frequency": "f",
        "id": 687,
        "name": "mask",
        "synonyms": ["mask", "facemask"],
        "synset": "mask.n.04"
    },
    {
        "def": "a vertical spar for supporting sails",
        "frequency": "f",
        "id": 688,
        "name": "mast",
        "synonyms": ["mast"],
        "synset": "mast.n.01"
    },
    {
        "def": "sports equipment consisting of a piece of thick padding on the floor for "
               "gymnastics",
        "frequency": "c",
        "id": 689,
        "name": "mat_(gym_equipment)",
        "synonyms": ["mat_(gym_equipment)", "gym_mat"],
        "synset": "mat.n.03"
    },
    {
        "def": "a box for holding matches",
        "frequency": "r",
        "id": 690,
        "name": "matchbox",
        "synonyms": ["matchbox"],
        "synset": "matchbox.n.01"
    },
    {
        "def": "a thick pad filled with resilient material used as a bed or part of a bed",
        "frequency": "f",
        "id": 691,
        "name": "mattress",
        "synonyms": ["mattress"],
        "synset": "mattress.n.01"
    },
    {
        "def": "graduated cup used to measure liquid or granular ingredients",
        "frequency": "c",
        "id": 692,
        "name": "measuring_cup",
        "synonyms": ["measuring_cup"],
        "synset": "measuring_cup.n.01"
    },
    {
        "def": "measuring instrument having a sequence of marks at regular intervals",
        "frequency": "c",
        "id": 693,
        "name": "measuring_stick",
        "synonyms": ["measuring_stick", "ruler_(measuring_stick)", "measuring_rod"],
        "synset": "measuring_stick.n.01"
    },
    {
        "def": "ground meat formed into a ball and fried or simmered in broth",
        "frequency": "c",
        "id": 694,
        "name": "meatball",
        "synonyms": ["meatball"],
        "synset": "meatball.n.01"
    },
    {
        "def": "something that treats or prevents or alleviates the symptoms of disease",
        "frequency": "c",
        "id": 695,
        "name": "medicine",
        "synonyms": ["medicine"],
        "synset": "medicine.n.02"
    },
    {
        "def": "fruit of the gourd family having a hard rind and sweet juicy flesh",
        "frequency": "r",
        "id": 696,
        "name": "melon",
        "synonyms": ["melon"],
        "synset": "melon.n.01"
    },
    {
        "def": "device for converting sound waves into electrical energy",
        "frequency": "f",
        "id": 697,
        "name": "microphone",
        "synonyms": ["microphone"],
        "synset": "microphone.n.01"
    },
    {
        "def": "magnifier of the image of small objects",
        "frequency": "r",
        "id": 698,
        "name": "microscope",
        "synonyms": ["microscope"],
        "synset": "microscope.n.01"
    },
    {
        "def": "kitchen appliance that cooks food by passing an electromagnetic wave through it",
        "frequency": "f",
        "id": 699,
        "name": "microwave_oven",
        "synonyms": ["microwave_oven"],
        "synset": "microwave.n.02"
    },
    {
        "def": "stone post at side of a road to show distances",
        "frequency": "r",
        "id": 700,
        "name": "milestone",
        "synonyms": ["milestone", "milepost"],
        "synset": "milestone.n.01"
    },
    {
        "def": "a white nutritious liquid secreted by mammals and used as food by human beings",
        "frequency": "c",
        "id": 701,
        "name": "milk",
        "synonyms": ["milk"],
        "synset": "milk.n.01"
    },
    {
        "def": "a small box-shaped passenger van",
        "frequency": "f",
        "id": 702,
        "name": "minivan",
        "synonyms": ["minivan"],
        "synset": "minivan.n.01"
    },
    {
        "def": "a candy that is flavored with a mint oil",
        "frequency": "r",
        "id": 703,
        "name": "mint_candy",
        "synonyms": ["mint_candy"],
        "synset": "mint.n.05"
    },
    {
        "def": "polished surface that forms images by reflecting light",
        "frequency": "f",
        "id": 704,
        "name": "mirror",
        "synonyms": ["mirror"],
        "synset": "mirror.n.01"
    },
    {
        "def": "glove that encases the thumb separately and the other four fingers together",
        "frequency": "c",
        "id": 705,
        "name": "mitten",
        "synonyms": ["mitten"],
        "synset": "mitten.n.01"
    },
    {
        "def": "a kitchen utensil that is used for mixing foods",
        "frequency": "c",
        "id": 706,
        "name": "mixer_(kitchen_tool)",
        "synonyms": ["mixer_(kitchen_tool)", "stand_mixer"],
        "synset": "mixer.n.04"
    },
    {
        "def": "the official currency issued by a government or national bank",
        "frequency": "c",
        "id": 707,
        "name": "money",
        "synonyms": ["money"],
        "synset": "money.n.03"
    },
    {
        "def": "a computer monitor",
        "frequency": "f",
        "id": 708,
        "name": "monitor_(computer_equipment) computer_monitor",
        "synonyms": ["monitor_(computer_equipment) computer_monitor"],
        "synset": "monitor.n.04"
    },
    {
        "def": "any of various long-tailed primates",
        "frequency": "c",
        "id": 709,
        "name": "monkey",
        "synonyms": ["monkey"],
        "synset": "monkey.n.01"
    },
    {
        "def": "machine that converts other forms of energy into mechanical energy and so imparts "
               "motion",
        "frequency": "f",
        "id": 710,
        "name": "motor",
        "synonyms": ["motor"],
        "synset": "motor.n.01"
    },
    {
        "def": "a wheeled vehicle with small wheels and a low-powered engine",
        "frequency": "f",
        "id": 711,
        "name": "motor_scooter",
        "synonyms": ["motor_scooter", "scooter"],
        "synset": "motor_scooter.n.01"
    },
    {
        "def": "a self-propelled wheeled vehicle that does not run on rails",
        "frequency": "r",
        "id": 712,
        "name": "motor_vehicle",
        "synonyms": ["motor_vehicle", "automotive_vehicle"],
        "synset": "motor_vehicle.n.01"
    },
    {
        "def": "a boat propelled by an internal-combustion engine",
        "frequency": "r",
        "id": 713,
        "name": "motorboat",
        "synonyms": ["motorboat", "powerboat"],
        "synset": "motorboat.n.01"
    },
    {
        "def": "a motor vehicle with two wheels and a strong frame",
        "frequency": "f",
        "id": 714,
        "name": "motorcycle",
        "synonyms": ["motorcycle"],
        "synset": "motorcycle.n.01"
    },
    {
        "def": "(baseball) the slight elevation on which the pitcher stands",
        "frequency": "f",
        "id": 715,
        "name": "mound_(baseball)",
        "synonyms": ["mound_(baseball)", "pitcher's_mound"],
        "synset": "mound.n.01"
    },
    {
        "def": "a small rodent with pointed snouts and small ears on elongated bodies with slender "
               "usually hairless tails",
        "frequency": "r",
        "id": 716,
        "name": "mouse_(animal_rodent)",
        "synonyms": ["mouse_(animal_rodent)"],
        "synset": "mouse.n.01"
    },
    {
        "def": "a computer input device that controls an on-screen pointer",
        "frequency": "f",
        "id": 717,
        "name": "mouse_(computer_equipment)",
        "synonyms": ["mouse_(computer_equipment)", "computer_mouse"],
        "synset": "mouse.n.04"
    },
    {
        "def": "a small portable pad that provides an operating surface for a computer mouse",
        "frequency": "f",
        "id": 718,
        "name": "mousepad",
        "synonyms": ["mousepad"],
        "synset": "mousepad.n.01"
    },
    {
        "def": "a sweet quick bread baked in a cup-shaped pan",
        "frequency": "c",
        "id": 719,
        "name": "muffin",
        "synonyms": ["muffin"],
        "synset": "muffin.n.01"
    },
    {
        "def": "with handle and usually cylindrical",
        "frequency": "f",
        "id": 720,
        "name": "mug",
        "synonyms": ["mug"],
        "synset": "mug.n.04"
    },
    {
        "def": "a common mushroom",
        "frequency": "f",
        "id": 721,
        "name": "mushroom",
        "synonyms": ["mushroom"],
        "synset": "mushroom.n.02"
    },
    {
        "def": "a stool for piano players; usually adjustable in height",
        "frequency": "r",
        "id": 722,
        "name": "music_stool",
        "synonyms": ["music_stool", "piano_stool"],
        "synset": "music_stool.n.01"
    },
    {
        "def": "any of various devices or contrivances that can be used to produce musical tones "
               "or sounds",
        "frequency": "r",
        "id": 723,
        "name": "musical_instrument",
        "synonyms": ["musical_instrument", "instrument_(musical)"],
        "synset": "musical_instrument.n.01"
    },
    {
        "def": "a small flat file for shaping the nails",
        "frequency": "r",
        "id": 724,
        "name": "nailfile",
        "synonyms": ["nailfile"],
        "synset": "nailfile.n.01"
    },
    {
        "def": "a plate bearing a name",
        "frequency": "r",
        "id": 725,
        "name": "nameplate",
        "synonyms": ["nameplate"],
        "synset": "nameplate.n.01"
    },
    {
        "def": "a small piece of table linen or paper that is used to wipe the mouth and to cover "
               "the lap in order to protect clothing",
        "frequency": "f",
        "id": 726,
        "name": "napkin",
        "synonyms": ["napkin", "table_napkin", "serviette"],
        "synset": "napkin.n.01"
    },
    {
        "def": "a kerchief worn around the neck",
        "frequency": "r",
        "id": 727,
        "name": "neckerchief",
        "synonyms": ["neckerchief"],
        "synset": "neckerchief.n.01"
    },
    {
        "def": "jewelry consisting of a cord or chain (often bearing gems) worn about the neck as "
               "an ornament",
        "frequency": "f",
        "id": 728,
        "name": "necklace",
        "synonyms": ["necklace"],
        "synset": "necklace.n.01"
    },
    {
        "def": "neckwear consisting of a long narrow piece of material worn under a collar and "
               "tied in knot at the front",
        "frequency": "f",
        "id": 729,
        "name": "necktie",
        "synonyms": ["necktie", "tie_(necktie)"],
        "synset": "necktie.n.01"
    },
    {
        "def": "a sharp pointed implement (usually metal)",
        "frequency": "r",
        "id": 730,
        "name": "needle",
        "synonyms": ["needle"],
        "synset": "needle.n.03"
    },
    {
        "def": "a structure in which animals lay eggs or give birth to their young",
        "frequency": "c",
        "id": 731,
        "name": "nest",
        "synonyms": ["nest"],
        "synset": "nest.n.01"
    },
    {
        "def": "a stall where newspapers and other periodicals are sold",
        "frequency": "r",
        "id": 732,
        "name": "newsstand",
        "synonyms": ["newsstand"],
        "synset": "newsstand.n.01"
    },
    {
        "def": "garments designed to be worn in bed",
        "frequency": "c",
        "id": 733,
        "name": "nightshirt",
        "synonyms": ["nightshirt", "nightwear", "sleepwear", "nightclothes"],
        "synset": "nightwear.n.01"
    },
    {
        "def": "a canvas bag that is used to feed an animal (such as a horse); covers the muzzle "
               "and fastens at the top of the head",
        "frequency": "r",
        "id": 734,
        "name": "nosebag_(for_animals)",
        "synonyms": ["nosebag_(for_animals)", "feedbag"],
        "synset": "nosebag.n.01"
    },
    {
        "def": "a strap that is the part of a bridle that goes over the animal's nose",
        "frequency": "r",
        "id": 735,
        "name": "noseband_(for_animals)",
        "synonyms": ["noseband_(for_animals)", "nosepiece_(for_animals)"],
        "synset": "noseband.n.01"
    },
    {
        "def": "a book with blank pages for recording notes or memoranda",
        "frequency": "f",
        "id": 736,
        "name": "notebook",
        "synonyms": ["notebook"],
        "synset": "notebook.n.01"
    },
    {
        "def": "a pad of paper for keeping notes",
        "frequency": "c",
        "id": 737,
        "name": "notepad",
        "synonyms": ["notepad"],
        "synset": "notepad.n.01"
    },
    {
        "def": "a small metal block (usually square or hexagonal) with internal screw thread to be "
               "fitted onto a bolt",
        "frequency": "c",
        "id": 738,
        "name": "nut",
        "synonyms": ["nut"],
        "synset": "nut.n.03"
    },
    {
        "def": "a hand tool used to crack nuts open",
        "frequency": "r",
        "id": 739,
        "name": "nutcracker",
        "synonyms": ["nutcracker"],
        "synset": "nutcracker.n.01"
    },
    {
        "def": "an implement used to propel or steer a boat",
        "frequency": "c",
        "id": 740,
        "name": "oar",
        "synonyms": ["oar"],
        "synset": "oar.n.01"
    },
    {
        "def": "tentacles of octopus prepared as food",
        "frequency": "r",
        "id": 741,
        "name": "octopus_(food)",
        "synonyms": ["octopus_(food)"],
        "synset": "octopus.n.01"
    },
    {
        "def": "bottom-living cephalopod having a soft oval body with eight long tentacles",
        "frequency": "r",
        "id": 742,
        "name": "octopus_(animal)",
        "synonyms": ["octopus_(animal)"],
        "synset": "octopus.n.02"
    },
    {
        "def": "a lamp that burns oil (as kerosine) for light",
        "frequency": "c",
        "id": 743,
        "name": "oil_lamp",
        "synonyms": ["oil_lamp", "kerosene_lamp", "kerosine_lamp"],
        "synset": "oil_lamp.n.01"
    },
    {
        "def": "oil from olives",
        "frequency": "c",
        "id": 744,
        "name": "olive_oil",
        "synonyms": ["olive_oil"],
        "synset": "olive_oil.n.01"
    },
    {
        "def": "beaten eggs cooked until just set; may be folded around e.g. ham or cheese or "
               "jelly",
        "frequency": "r",
        "id": 745,
        "name": "omelet",
        "synonyms": ["omelet", "omelette"],
        "synset": "omelet.n.01"
    },
    {
        "def": "the bulb of an onion plant",
        "frequency": "f",
        "id": 746,
        "name": "onion",
        "synonyms": ["onion"],
        "synset": "onion.n.01"
    },
    {
        "def": "orange (FRUIT of an orange tree)",
        "frequency": "f",
        "id": 747,
        "name": "orange_(fruit)",
        "synonyms": ["orange_(fruit)"],
        "synset": "orange.n.01"
    },
    {
        "def": "bottled or freshly squeezed juice of oranges",
        "frequency": "c",
        "id": 748,
        "name": "orange_juice",
        "synonyms": ["orange_juice"],
        "synset": "orange_juice.n.01"
    },
    {
        "def": "aromatic Eurasian perennial herb used in cooking and baking",
        "frequency": "r",
        "id": 749,
        "name": "oregano",
        "synonyms": ["oregano", "marjoram"],
        "synset": "oregano.n.01"
    },
    {
        "def": "fast-running African flightless bird with two-toed feet; largest living bird",
        "frequency": "c",
        "id": 750,
        "name": "ostrich",
        "synonyms": ["ostrich"],
        "synset": "ostrich.n.02"
    },
    {
        "def": "thick cushion used as a seat",
        "frequency": "c",
        "id": 751,
        "name": "ottoman",
        "synonyms": ["ottoman", "pouf", "pouffe", "hassock"],
        "synset": "ottoman.n.03"
    },
    {
        "def": "work clothing consisting of denim trousers usually with a bib and shoulder straps",
        "frequency": "c",
        "id": 752,
        "name": "overalls_(clothing)",
        "synonyms": ["overalls_(clothing)"],
        "synset": "overall.n.01"
    },
    {
        "def": "nocturnal bird of prey with hawk-like beak and claws and large head with "
               "front-facing eyes",
        "frequency": "c",
        "id": 753,
        "name": "owl",
        "synonyms": ["owl"],
        "synset": "owl.n.01"
    },
    {
        "def": "a small package or bundle",
        "frequency": "c",
        "id": 754,
        "name": "packet",
        "synonyms": ["packet"],
        "synset": "packet.n.03"
    },
    {
        "def": "absorbent material saturated with ink used to transfer ink evenly to a rubber "
               "stamp",
        "frequency": "r",
        "id": 755,
        "name": "inkpad",
        "synonyms": ["inkpad", "inking_pad", "stamp_pad"],
        "synset": "pad.n.03"
    },
    {
        "def": "a flat mass of soft material used for protection, stuffing, or comfort",
        "frequency": "c",
        "id": 756,
        "name": "pad",
        "synonyms": ["pad"],
        "synset": "pad.n.04"
    },
    {
        "def": "a short light oar used without an oarlock to propel a canoe or small boat",
        "frequency": "c",
        "id": 757,
        "name": "paddle",
        "synonyms": ["paddle", "boat_paddle"],
        "synset": "paddle.n.04"
    },
    {
        "def": "a detachable, portable lock",
        "frequency": "c",
        "id": 758,
        "name": "padlock",
        "synonyms": ["padlock"],
        "synset": "padlock.n.01"
    },
    {
        "def": "a box containing a collection of cubes or tubes of artists' paint",
        "frequency": "r",
        "id": 759,
        "name": "paintbox",
        "synonyms": ["paintbox"],
        "synset": "paintbox.n.01"
    },
    {
        "def": "a brush used as an applicator to apply paint",
        "frequency": "c",
        "id": 760,
        "name": "paintbrush",
        "synonyms": ["paintbrush"],
        "synset": "paintbrush.n.01"
    },
    {
        "def": "graphic art consisting of an artistic composition made by applying paints to a "
               "surface",
        "frequency": "f",
        "id": 761,
        "name": "painting",
        "synonyms": ["painting"],
        "synset": "painting.n.01"
    },
    {
        "def": "loose-fitting nightclothes worn for sleeping or lounging",
        "frequency": "c",
        "id": 762,
        "name": "pajamas",
        "synonyms": ["pajamas", "pyjamas"],
        "synset": "pajama.n.02"
    },
    {
        "def": "board that provides a flat surface on which artists mix paints and the range of "
               "colors used",
        "frequency": "c",
        "id": 763,
        "name": "palette",
        "synonyms": ["palette", "pallet"],
        "synset": "palette.n.02"
    },
    {
        "def": "cooking utensil consisting of a wide metal vessel",
        "frequency": "f",
        "id": 764,
        "name": "pan_(for_cooking)",
        "synonyms": ["pan_(for_cooking)", "cooking_pan"],
        "synset": "pan.n.01"
    },
    {
        "def": "shallow container made of metal",
        "frequency": "r",
        "id": 765,
        "name": "pan_(metal_container)",
        "synonyms": ["pan_(metal_container)"],
        "synset": "pan.n.03"
    },
    {
        "def": "a flat cake of thin batter fried on both sides on a griddle",
        "frequency": "c",
        "id": 766,
        "name": "pancake",
        "synonyms": ["pancake"],
        "synset": "pancake.n.01"
    },
    {
        "def": "a woman's tights consisting of underpants and stockings",
        "frequency": "r",
        "id": 767,
        "name": "pantyhose",
        "synonyms": ["pantyhose"],
        "synset": "pantyhose.n.01"
    },
    {
        "def": "large oval melon-like tropical fruit with yellowish flesh",
        "frequency": "r",
        "id": 768,
        "name": "papaya",
        "synonyms": ["papaya"],
        "synset": "papaya.n.02"
    },
    {
        "def": "a wire or plastic clip for holding sheets of paper together",
        "frequency": "r",
        "id": 769,
        "name": "paperclip",
        "synonyms": ["paperclip"],
        "synset": "paper_clip.n.01"
    },
    {
        "def": "a disposable plate made of cardboard",
        "frequency": "f",
        "id": 770,
        "name": "paper_plate",
        "synonyms": ["paper_plate"],
        "synset": "paper_plate.n.01"
    },
    {
        "def": "a disposable towel made of absorbent paper",
        "frequency": "f",
        "id": 771,
        "name": "paper_towel",
        "synonyms": ["paper_towel"],
        "synset": "paper_towel.n.01"
    },
    {
        "def": "a book with paper covers",
        "frequency": "r",
        "id": 772,
        "name": "paperback_book",
        "synonyms": ["paperback_book", "paper-back_book", "softback_book", "soft-cover_book"],
        "synset": "paperback_book.n.01"
    },
    {
        "def": "a weight used to hold down a stack of papers",
        "frequency": "r",
        "id": 773,
        "name": "paperweight",
        "synonyms": ["paperweight"],
        "synset": "paperweight.n.01"
    },
    {
        "def": "rescue equipment consisting of a device that fills with air and retards your fall",
        "frequency": "c",
        "id": 774,
        "name": "parachute",
        "synonyms": ["parachute"],
        "synset": "parachute.n.01"
    },
    {
        "def": "any of numerous small slender long-tailed parrots",
        "frequency": "r",
        "id": 775,
        "name": "parakeet",
        "synonyms": ["parakeet", "parrakeet", "parroket", "paraquet", "paroquet", "parroquet"],
        "synset": "parakeet.n.01"
    },
    {
        "def": "parachute that will lift a person up into the air when it is towed by a motorboat "
               "or a car",
        "frequency": "c",
        "id": 776,
        "name": "parasail_(sports)",
        "synonyms": ["parasail_(sports)"],
        "synset": "parasail.n.01"
    },
    {
        "def": "a superior paper resembling sheepskin",
        "frequency": "r",
        "id": 777,
        "name": "parchment",
        "synonyms": ["parchment"],
        "synset": "parchment.n.01"
    },
    {
        "def": "a kind of heavy jacket (`windcheater' is a British term)",
        "frequency": "r",
        "id": 778,
        "name": "parka",
        "synonyms": ["parka", "anorak"],
        "synset": "parka.n.01"
    },
    {
        "def": "a coin-operated timer located next to a parking space",
        "frequency": "f",
        "id": 779,
        "name": "parking_meter",
        "synonyms": ["parking_meter"],
        "synset": "parking_meter.n.01"
    },
    {
        "def": "usually brightly colored tropical birds with short hooked beaks and the ability to "
               "mimic sounds",
        "frequency": "c",
        "id": 780,
        "name": "parrot",
        "synonyms": ["parrot"],
        "synset": "parrot.n.01"
    },
    {
        "def": "a railcar where passengers ride",
        "frequency": "c",
        "id": 781,
        "name": "passenger_car_(part_of_a_train)",
        "synonyms": ["passenger_car_(part_of_a_train)", "coach_(part_of_a_train)"],
        "synset": "passenger_car.n.01"
    },
    {
        "def": "a ship built to carry passengers",
        "frequency": "r",
        "id": 782,
        "name": "passenger_ship",
        "synonyms": ["passenger_ship"],
        "synset": "passenger_ship.n.01"
    },
    {
        "def": "a document issued by a country to a citizen allowing that person to travel abroad "
               "and re-enter the home country",
        "frequency": "r",
        "id": 783,
        "name": "passport",
        "synonyms": ["passport"],
        "synset": "passport.n.02"
    },
    {
        "def": "any of various baked foods made of dough or batter",
        "frequency": "f",
        "id": 784,
        "name": "pastry",
        "synonyms": ["pastry"],
        "synset": "pastry.n.02"
    },
    {
        "def": "small flat mass of chopped food",
        "frequency": "r",
        "id": 785,
        "name": "patty_(food)",
        "synonyms": ["patty_(food)"],
        "synset": "patty.n.01"
    },
    {
        "def": "seed of a pea plant used for food",
        "frequency": "c",
        "id": 786,
        "name": "pea_(food)",
        "synonyms": ["pea_(food)"],
        "synset": "pea.n.01"
    },
    {
        "def": "downy juicy fruit with sweet yellowish or whitish flesh",
        "frequency": "c",
        "id": 787,
        "name": "peach",
        "synonyms": ["peach"],
        "synset": "peach.n.03"
    },
    {
        "def": "a spread made from ground peanuts",
        "frequency": "c",
        "id": 788,
        "name": "peanut_butter",
        "synonyms": ["peanut_butter"],
        "synset": "peanut_butter.n.01"
    },
    {
        "def": "sweet juicy gritty-textured fruit available in many varieties",
        "frequency": "c",
        "id": 789,
        "name": "pear",
        "synonyms": ["pear"],
        "synset": "pear.n.01"
    },
    {
        "def": "a device for peeling vegetables or fruits",
        "frequency": "r",
        "id": 790,
        "name": "peeler_(tool_for_fruit_and_vegetables)",
        "synonyms": ["peeler_(tool_for_fruit_and_vegetables)"],
        "synset": "peeler.n.03"
    },
    {
        "def": "a board perforated with regularly spaced holes into which pegs can be fitted",
        "frequency": "r",
        "id": 791,
        "name": "pegboard",
        "synonyms": ["pegboard"],
        "synset": "pegboard.n.01"
    },
    {
        "def": "large long-winged warm-water seabird having a large bill with a distensible pouch "
               "for fish",
        "frequency": "c",
        "id": 792,
        "name": "pelican",
        "synonyms": ["pelican"],
        "synset": "pelican.n.01"
    },
    {
        "def": "a writing implement with a point from which ink flows",
        "frequency": "f",
        "id": 793,
        "name": "pen",
        "synonyms": ["pen"],
        "synset": "pen.n.01"
    },
    {
        "def": "a thin cylindrical pointed writing implement made of wood and graphite",
        "frequency": "c",
        "id": 794,
        "name": "pencil",
        "synonyms": ["pencil"],
        "synset": "pencil.n.01"
    },
    {
        "def": "a box for holding pencils",
        "frequency": "r",
        "id": 795,
        "name": "pencil_box",
        "synonyms": ["pencil_box", "pencil_case"],
        "synset": "pencil_box.n.01"
    },
    {
        "def": "a rotary implement for sharpening the point on pencils",
        "frequency": "r",
        "id": 796,
        "name": "pencil_sharpener",
        "synonyms": ["pencil_sharpener"],
        "synset": "pencil_sharpener.n.01"
    },
    {
        "def": "an apparatus consisting of an object mounted so that it swings freely under the "
               "influence of gravity",
        "frequency": "r",
        "id": 797,
        "name": "pendulum",
        "synonyms": ["pendulum"],
        "synset": "pendulum.n.01"
    },
    {
        "def": "short-legged flightless birds of cold southern regions having webbed feet and "
               "wings modified as flippers",
        "frequency": "c",
        "id": 798,
        "name": "penguin",
        "synonyms": ["penguin"],
        "synset": "penguin.n.01"
    },
    {
        "def": "a flag longer than it is wide (and often tapering)",
        "frequency": "r",
        "id": 799,
        "name": "pennant",
        "synonyms": ["pennant"],
        "synset": "pennant.n.02"
    },
    {
        "def": "a coin worth one-hundredth of the value of the basic unit",
        "frequency": "r",
        "id": 800,
        "name": "penny_(coin)",
        "synonyms": ["penny_(coin)"],
        "synset": "penny.n.02"
    },
    {
        "def": "pungent seasoning from the berry of the common pepper plant; whole or ground",
        "frequency": "c",
        "id": 801,
        "name": "pepper",
        "synonyms": ["pepper", "peppercorn"],
        "synset": "pepper.n.03"
    },
    {
        "def": "a mill for grinding pepper",
        "frequency": "c",
        "id": 802,
        "name": "pepper_mill",
        "synonyms": ["pepper_mill", "pepper_grinder"],
        "synset": "pepper_mill.n.01"
    },
    {
        "def": "a toiletry that emits and diffuses a fragrant odor",
        "frequency": "c",
        "id": 803,
        "name": "perfume",
        "synonyms": ["perfume"],
        "synset": "perfume.n.02"
    },
    {
        "def": "orange fruit resembling a plum; edible when fully ripe",
        "frequency": "r",
        "id": 804,
        "name": "persimmon",
        "synonyms": ["persimmon"],
        "synset": "persimmon.n.02"
    },
    {
        "def": "a human being",
        "frequency": "f",
        "id": 805,
        "name": "baby",
        "synonyms": ["baby", "child", "boy", "girl", "man", "woman", "person", "human"],
        "synset": "person.n.01"
    },
    {
        "def": "a domesticated animal kept for companionship or amusement",
        "frequency": "r",
        "id": 806,
        "name": "pet",
        "synonyms": ["pet"],
        "synset": "pet.n.01"
    },
    {
        "def": "food prepared for animal pets",
        "frequency": "r",
        "id": 807,
        "name": "petfood",
        "synonyms": ["petfood", "pet-food"],
        "synset": "petfood.n.01"
    },
    {
        "def": "long bench with backs; used in church by the congregation",
        "frequency": "r",
        "id": 808,
        "name": "pew_(church_bench)",
        "synonyms": ["pew_(church_bench)", "church_bench"],
        "synset": "pew.n.01"
    },
    {
        "def": "a directory containing an alphabetical list of telephone subscribers and their "
               "telephone numbers",
        "frequency": "r",
        "id": 809,
        "name": "phonebook",
        "synonyms": ["phonebook", "telephone_book", "telephone_directory"],
        "synset": "phonebook.n.01"
    },
    {
        "def": "sound recording consisting of a typically black disk with a continuous groove",
        "frequency": "c",
        "id": 810,
        "name": "phonograph_record",
        "synonyms": ["phonograph_record", "phonograph_recording", "record_(phonograph_recording)"],
        "synset": "phonograph_record.n.01"
    },
    {
        "def": "a keyboard instrument that is played by depressing keys that cause hammers to "
               "strike tuned strings and produce sounds",
        "frequency": "c",
        "id": 811,
        "name": "piano",
        "synonyms": ["piano"],
        "synset": "piano.n.01"
    },
    {
        "def": "vegetables (especially cucumbers) preserved in brine or vinegar",
        "frequency": "f",
        "id": 812,
        "name": "pickle",
        "synonyms": ["pickle"],
        "synset": "pickle.n.01"
    },
    {
        "def": "a light truck with an open body and low sides and a tailboard",
        "frequency": "f",
        "id": 813,
        "name": "pickup_truck",
        "synonyms": ["pickup_truck"],
        "synset": "pickup.n.01"
    },
    {
        "def": "dish baked in pastry-lined pan often with a pastry top",
        "frequency": "c",
        "id": 814,
        "name": "pie",
        "synonyms": ["pie"],
        "synset": "pie.n.01"
    },
    {
        "def": "wild and domesticated birds having a heavy body and short legs",
        "frequency": "c",
        "id": 815,
        "name": "pigeon",
        "synonyms": ["pigeon"],
        "synset": "pigeon.n.01"
    },
    {
        "def": "a child's coin bank (often shaped like a pig)",
        "frequency": "r",
        "id": 816,
        "name": "piggy_bank",
        "synonyms": ["piggy_bank", "penny_bank"],
        "synset": "piggy_bank.n.01"
    },
    {
        "def": "a cushion to support the head of a sleeping person",
        "frequency": "f",
        "id": 817,
        "name": "pillow",
        "synonyms": ["pillow"],
        "synset": "pillow.n.01"
    },
    {
        "def": "a small slender (often pointed) piece of wood or metal used to support or fasten "
               "or attach things",
        "frequency": "r",
        "id": 818,
        "name": "pin_(non_jewelry)",
        "synonyms": ["pin_(non_jewelry)"],
        "synset": "pin.n.09"
    },
    {
        "def": "large sweet fleshy tropical fruit with a tuft of stiff leaves",
        "frequency": "f",
        "id": 819,
        "name": "pineapple",
        "synonyms": ["pineapple"],
        "synset": "pineapple.n.02"
    },
    {
        "def": "the seed-producing cone of a pine tree",
        "frequency": "c",
        "id": 820,
        "name": "pinecone",
        "synonyms": ["pinecone"],
        "synset": "pinecone.n.01"
    },
    {
        "def": "light hollow ball used in playing table tennis",
        "frequency": "r",
        "id": 821,
        "name": "ping-pong_ball",
        "synonyms": ["ping-pong_ball"],
        "synset": "ping-pong_ball.n.01"
    },
    {
        "def": "a toy consisting of vanes of colored paper or plastic that is pinned to a stick "
               "and spins when it is pointed into the wind",
        "frequency": "r",
        "id": 822,
        "name": "pinwheel",
        "synonyms": ["pinwheel"],
        "synset": "pinwheel.n.03"
    },
    {
        "def": "a tube with a small bowl at one end; used for smoking tobacco",
        "frequency": "r",
        "id": 823,
        "name": "tobacco_pipe",
        "synonyms": ["tobacco_pipe"],
        "synset": "pipe.n.01"
    },
    {
        "def": "a long tube made of metal or plastic that is used to carry water or oil or gas "
               "etc.",
        "frequency": "f",
        "id": 824,
        "name": "pipe",
        "synonyms": ["pipe", "piping"],
        "synset": "pipe.n.02"
    },
    {
        "def": "a firearm that is held and fired with one hand",
        "frequency": "r",
        "id": 825,
        "name": "pistol",
        "synonyms": ["pistol", "handgun"],
        "synset": "pistol.n.01"
    },
    {
        "def": "usually small round bread that can open into a pocket for filling",
        "frequency": "r",
        "id": 826,
        "name": "pita_(bread)",
        "synonyms": ["pita_(bread)", "pocket_bread"],
        "synset": "pita.n.01"
    },
    {
        "def": "an open vessel with a handle and a spout for pouring",
        "frequency": "f",
        "id": 827,
        "name": "pitcher_(vessel_for_liquid)",
        "synonyms": ["pitcher_(vessel_for_liquid)", "ewer"],
        "synset": "pitcher.n.02"
    },
    {
        "def": "a long-handled hand tool with sharp widely spaced prongs for lifting and pitching "
               "hay",
        "frequency": "r",
        "id": 828,
        "name": "pitchfork",
        "synonyms": ["pitchfork"],
        "synset": "pitchfork.n.01"
    },
    {
        "def": "Italian open pie made of thin bread dough spread with a spiced mixture of e.g. "
               "tomato sauce and cheese",
        "frequency": "f",
        "id": 829,
        "name": "pizza",
        "synonyms": ["pizza"],
        "synset": "pizza.n.01"
    },
    {
        "def": "a mat placed on a table for an individual place setting",
        "frequency": "f",
        "id": 830,
        "name": "place_mat",
        "synonyms": ["place_mat"],
        "synset": "place_mat.n.01"
    },
    {
        "def": "dish on which food is served or from which food is eaten",
        "frequency": "f",
        "id": 831,
        "name": "plate",
        "synonyms": ["plate"],
        "synset": "plate.n.04"
    },
    {
        "def": "a large shallow dish used for serving food",
        "frequency": "c",
        "id": 832,
        "name": "platter",
        "synonyms": ["platter"],
        "synset": "platter.n.01"
    },
    {
        "def": "one of a pack of cards that are used to play card games",
        "frequency": "r",
        "id": 833,
        "name": "playing_card",
        "synonyms": ["playing_card"],
        "synset": "playing_card.n.01"
    },
    {
        "def": "a portable enclosure in which babies may be left to play",
        "frequency": "r",
        "id": 834,
        "name": "playpen",
        "synonyms": ["playpen"],
        "synset": "playpen.n.01"
    },
    {
        "def": "a gripping hand tool with two hinged arms and (usually) serrated jaws",
        "frequency": "c",
        "id": 835,
        "name": "pliers",
        "synonyms": ["pliers", "plyers"],
        "synset": "pliers.n.01"
    },
    {
        "def": "a farm tool having one or more heavy blades to break the soil and cut a furrow "
               "prior to sowing",
        "frequency": "r",
        "id": 836,
        "name": "plow_(farm_equipment)",
        "synonyms": ["plow_(farm_equipment)", "plough_(farm_equipment)"],
        "synset": "plow.n.01"
    },
    {
        "def": "a watch that is carried in a small watch pocket",
        "frequency": "r",
        "id": 837,
        "name": "pocket_watch",
        "synonyms": ["pocket_watch"],
        "synset": "pocket_watch.n.01"
    },
    {
        "def": "a knife with a blade that folds into the handle; suitable for carrying in the "
               "pocket",
        "frequency": "c",
        "id": 838,
        "name": "pocketknife",
        "synonyms": ["pocketknife"],
        "synset": "pocketknife.n.01"
    },
    {
        "def": "fire iron consisting of a metal rod with a handle; used to stir a fire",
        "frequency": "c",
        "id": 839,
        "name": "poker_(fire_stirring_tool)",
        "synonyms": ["poker_(fire_stirring_tool)", "stove_poker", "fire_hook"],
        "synset": "poker.n.01"
    },
    {
        "def": "a long (usually round) rod of wood or metal or plastic",
        "frequency": "f",
        "id": 840,
        "name": "pole",
        "synonyms": ["pole", "post"],
        "synset": "pole.n.01"
    },
    {
        "def": "van used by police to transport prisoners",
        "frequency": "r",
        "id": 841,
        "name": "police_van",
        "synonyms": ["police_van", "police_wagon", "paddy_wagon", "patrol_wagon"],
        "synset": "police_van.n.01"
    },
    {
        "def": "a shirt with short sleeves designed for comfort and casual wear",
        "frequency": "f",
        "id": 842,
        "name": "polo_shirt",
        "synonyms": ["polo_shirt", "sport_shirt"],
        "synset": "polo_shirt.n.01"
    },
    {
        "def": "a blanket-like cloak with a hole in the center for the head",
        "frequency": "r",
        "id": 843,
        "name": "poncho",
        "synonyms": ["poncho"],
        "synset": "poncho.n.01"
    },
    {
        "def": "any of various breeds of small gentle horses usually less than five feet high at "
               "the shoulder",
        "frequency": "c",
        "id": 844,
        "name": "pony",
        "synonyms": ["pony"],
        "synset": "pony.n.05"
    },
    {
        "def": "game equipment consisting of a heavy table on which pool is played",
        "frequency": "r",
        "id": 845,
        "name": "pool_table",
        "synonyms": ["pool_table", "billiard_table", "snooker_table"],
        "synset": "pool_table.n.01"
    },
    {
        "def": "a sweet drink containing carbonated water and flavoring",
        "frequency": "f",
        "id": 846,
        "name": "pop_(soda)",
        "synonyms": ["pop_(soda)", "soda_(pop)", "tonic", "soft_drink"],
        "synset": "pop.n.02"
    },
    {
        "def": "any likeness of a person, in any medium",
        "frequency": "r",
        "id": 847,
        "name": "portrait",
        "synonyms": ["portrait", "portrayal"],
        "synset": "portrait.n.02"
    },
    {
        "def": "public box for deposit of mail",
        "frequency": "c",
        "id": 848,
        "name": "postbox_(public)",
        "synonyms": ["postbox_(public)", "mailbox_(public)"],
        "synset": "postbox.n.01"
    },
    {
        "def": "a card for sending messages by post without an envelope",
        "frequency": "c",
        "id": 849,
        "name": "postcard",
        "synonyms": ["postcard", "postal_card", "mailing-card"],
        "synset": "postcard.n.01"
    },
    {
        "def": "a sign posted in a public place as an advertisement",
        "frequency": "f",
        "id": 850,
        "name": "poster",
        "synonyms": ["poster", "placard"],
        "synset": "poster.n.01"
    },
    {
        "def": "metal or earthenware cooking vessel that is usually round and deep; often has a "
               "handle and lid",
        "frequency": "f",
        "id": 851,
        "name": "pot",
        "synonyms": ["pot"],
        "synset": "pot.n.01"
    },
    {
        "def": "a container in which plants are cultivated",
        "frequency": "f",
        "id": 852,
        "name": "flowerpot",
        "synonyms": ["flowerpot"],
        "synset": "pot.n.04"
    },
    {
        "def": "an edible tuber native to South America",
        "frequency": "f",
        "id": 853,
        "name": "potato",
        "synonyms": ["potato"],
        "synset": "potato.n.01"
    },
    {
        "def": "an insulated pad for holding hot pots",
        "frequency": "c",
        "id": 854,
        "name": "potholder",
        "synonyms": ["potholder"],
        "synset": "potholder.n.01"
    },
    {
        "def": "ceramic ware made from clay and baked in a kiln",
        "frequency": "c",
        "id": 855,
        "name": "pottery",
        "synonyms": ["pottery", "clayware"],
        "synset": "pottery.n.01"
    },
    {
        "def": "a small or medium size container for holding or carrying things",
        "frequency": "c",
        "id": 856,
        "name": "pouch",
        "synonyms": ["pouch"],
        "synset": "pouch.n.01"
    },
    {
        "def": "a machine for excavating",
        "frequency": "r",
        "id": 857,
        "name": "power_shovel",
        "synonyms": ["power_shovel", "excavator", "digger"],
        "synset": "power_shovel.n.01"
    },
    {
        "def": "any of various edible decapod crustaceans",
        "frequency": "c",
        "id": 858,
        "name": "prawn",
        "synonyms": ["prawn", "shrimp"],
        "synset": "prawn.n.01"
    },
    {
        "def": "a machine that prints",
        "frequency": "f",
        "id": 859,
        "name": "printer",
        "synonyms": ["printer", "printing_machine"],
        "synset": "printer.n.03"
    },
    {
        "def": "a weapon that is forcibly thrown or projected at a targets",
        "frequency": "c",
        "id": 860,
        "name": "projectile_(weapon)",
        "synonyms": ["projectile_(weapon)", "missile"],
        "synset": "projectile.n.01"
    },
    {
        "def": "an optical instrument that projects an enlarged image onto a screen",
        "frequency": "c",
        "id": 861,
        "name": "projector",
        "synonyms": ["projector"],
        "synset": "projector.n.02"
    },
    {
        "def": "a mechanical device that rotates to push against air or water",
        "frequency": "f",
        "id": 862,
        "name": "propeller",
        "synonyms": ["propeller", "propellor"],
        "synset": "propeller.n.01"
    },
    {
        "def": "dried plum",
        "frequency": "r",
        "id": 863,
        "name": "prune",
        "synonyms": ["prune"],
        "synset": "prune.n.01"
    },
    {
        "def": "any of various soft thick unsweetened baked dishes",
        "frequency": "r",
        "id": 864,
        "name": "pudding",
        "synonyms": ["pudding"],
        "synset": "pudding.n.01"
    },
    {
        "def": "fishes whose elongated spiny body can inflate itself with water or air to form a "
               "globe",
        "frequency": "r",
        "id": 865,
        "name": "puffer_(fish)",
        "synonyms": ["puffer_(fish)", "pufferfish", "blowfish", "globefish"],
        "synset": "puffer.n.02"
    },
    {
        "def": "seabirds having short necks and brightly colored compressed bills",
        "frequency": "r",
        "id": 866,
        "name": "puffin",
        "synonyms": ["puffin"],
        "synset": "puffin.n.01"
    },
    {
        "def": "small compact smooth-coated breed of Asiatic origin having a tightly curled tail "
               "and broad flat wrinkled muzzle",
        "frequency": "r",
        "id": 867,
        "name": "pug-dog",
        "synonyms": ["pug-dog"],
        "synset": "pug.n.01"
    },
    {
        "def": "usually large pulpy deep-yellow round fruit of the squash family maturing in late "
               "summer or early autumn",
        "frequency": "c",
        "id": 868,
        "name": "pumpkin",
        "synonyms": ["pumpkin"],
        "synset": "pumpkin.n.02"
    },
    {
        "def": "a tool for making holes or indentations",
        "frequency": "r",
        "id": 869,
        "name": "puncher",
        "synonyms": ["puncher"],
        "synset": "punch.n.03"
    },
    {
        "def": "a small figure of a person operated from above with strings by a puppeteer",
        "frequency": "r",
        "id": 870,
        "name": "puppet",
        "synonyms": ["puppet", "marionette"],
        "synset": "puppet.n.01"
    },
    {
        "def": "a young dog",
        "frequency": "r",
        "id": 871,
        "name": "puppy",
        "synonyms": ["puppy"],
        "synset": "puppy.n.01"
    },
    {
        "def": "a tortilla that is filled with cheese and heated",
        "frequency": "r",
        "id": 872,
        "name": "quesadilla",
        "synonyms": ["quesadilla"],
        "synset": "quesadilla.n.01"
    },
    {
        "def": "a tart filled with rich unsweetened custard; often contains other ingredients (as "
               "cheese or ham or seafood or vegetables)",
        "frequency": "r",
        "id": 873,
        "name": "quiche",
        "synonyms": ["quiche"],
        "synset": "quiche.n.02"
    },
    {
        "def": "bedding made of two layers of cloth filled with stuffing and stitched together",
        "frequency": "f",
        "id": 874,
        "name": "quilt",
        "synonyms": ["quilt", "comforter"],
        "synset": "quilt.n.01"
    },
    {
        "def": "any of various burrowing animals of the family Leporidae having long ears and "
               "short tails",
        "frequency": "c",
        "id": 875,
        "name": "rabbit",
        "synonyms": ["rabbit"],
        "synset": "rabbit.n.01"
    },
    {
        "def": "a fast car that competes in races",
        "frequency": "r",
        "id": 876,
        "name": "race_car",
        "synonyms": ["race_car", "racing_car"],
        "synset": "racer.n.02"
    },
    {
        "def": "a sports implement used to strike a ball in various games",
        "frequency": "c",
        "id": 877,
        "name": "racket",
        "synonyms": ["racket", "racquet"],
        "synset": "racket.n.04"
    },
    {
        "def": "measuring instrument in which the echo of a pulse of microwave radiation is used "
               "to detect and locate distant objects",
        "frequency": "r",
        "id": 878,
        "name": "radar",
        "synonyms": ["radar"],
        "synset": "radar.n.01"
    },
    {
        "def": "a mechanism consisting of a metal honeycomb through which hot fluids circulate",
        "frequency": "c",
        "id": 879,
        "name": "radiator",
        "synonyms": ["radiator"],
        "synset": "radiator.n.03"
    },
    {
        "def": "an electronic receiver that detects and demodulates and amplifies transmitted "
               "radio signals",
        "frequency": "c",
        "id": 880,
        "name": "radio_receiver",
        "synonyms": ["radio_receiver", "radio_set", "radio", "tuner_(radio)"],
        "synset": "radio_receiver.n.01"
    },
    {
        "def": "pungent edible root of any of various cultivated radish plants",
        "frequency": "c",
        "id": 881,
        "name": "radish",
        "synonyms": ["radish", "daikon"],
        "synset": "radish.n.03"
    },
    {
        "def": "a flat float (usually made of logs or planks) that can be used for transport or as "
               "a platform for swimmers",
        "frequency": "c",
        "id": 882,
        "name": "raft",
        "synonyms": ["raft"],
        "synset": "raft.n.01"
    },
    {
        "def": "a cloth doll that is stuffed and (usually) painted",
        "frequency": "r",
        "id": 883,
        "name": "rag_doll",
        "synonyms": ["rag_doll"],
        "synset": "rag_doll.n.01"
    },
    {
        "def": "a water-resistant coat",
        "frequency": "c",
        "id": 884,
        "name": "raincoat",
        "synonyms": ["raincoat", "waterproof_jacket"],
        "synset": "raincoat.n.01"
    },
    {
        "def": "uncastrated adult male sheep",
        "frequency": "c",
        "id": 885,
        "name": "ram_(animal)",
        "synonyms": ["ram_(animal)"],
        "synset": "ram.n.05"
    },
    {
        "def": "red or black edible aggregate berries usually smaller than the related "
               "blackberries",
        "frequency": "c",
        "id": 886,
        "name": "raspberry",
        "synonyms": ["raspberry"],
        "synset": "raspberry.n.02"
    },
    {
        "def": "any of various long-tailed rodents similar to but larger than a mouse",
        "frequency": "r",
        "id": 887,
        "name": "rat",
        "synonyms": ["rat"],
        "synset": "rat.n.01"
    },
    {
        "def": "a blade that has very sharp edge",
        "frequency": "c",
        "id": 888,
        "name": "razorblade",
        "synonyms": ["razorblade"],
        "synset": "razorblade.n.01"
    },
    {
        "def": "a squeezer with a conical ridged center that is used for squeezing juice from "
               "citrus fruit",
        "frequency": "c",
        "id": 889,
        "name": "reamer_(juicer)",
        "synonyms": ["reamer_(juicer)", "juicer", "juice_reamer"],
        "synset": "reamer.n.01"
    },
    {
        "def": "car mirror that reflects the view out of the rear window",
        "frequency": "f",
        "id": 890,
        "name": "rearview_mirror",
        "synonyms": ["rearview_mirror"],
        "synset": "rearview_mirror.n.01"
    },
    {
        "def": "an acknowledgment (usually tangible) that payment has been made",
        "frequency": "c",
        "id": 891,
        "name": "receipt",
        "synonyms": ["receipt"],
        "synset": "receipt.n.02"
    },
    {
        "def": "an armchair whose back can be lowered and foot can be raised to allow the sitter "
               "to recline in it",
        "frequency": "c",
        "id": 892,
        "name": "recliner",
        "synonyms": ["recliner", "reclining_chair", "lounger_(chair)"],
        "synset": "recliner.n.01"
    },
    {
        "def": "machine in which rotating records cause a stylus to vibrate and the vibrations are "
               "amplified acoustically or electronically",
        "frequency": "r",
        "id": 893,
        "name": "record_player",
        "synonyms": ["record_player", "phonograph_(record_player)", "turntable"],
        "synset": "record_player.n.01"
    },
    {
        "def": "compact head of purplish-red leaves",
        "frequency": "r",
        "id": 894,
        "name": "red_cabbage",
        "synonyms": ["red_cabbage"],
        "synset": "red_cabbage.n.02"
    },
    {
        "def": "device that reflects light, radiation, etc.",
        "frequency": "f",
        "id": 895,
        "name": "reflector",
        "synonyms": ["reflector"],
        "synset": "reflector.n.01"
    },
    {
        "def": "a device that can be used to control a machine or apparatus from a distance",
        "frequency": "f",
        "id": 896,
        "name": "remote_control",
        "synonyms": ["remote_control"],
        "synset": "remote_control.n.01"
    },
    {
        "def": "massive powerful herbivorous odd-toed ungulate of southeast Asia and Africa having "
               "very thick skin and one or two horns on the snout",
        "frequency": "c",
        "id": 897,
        "name": "rhinoceros",
        "synonyms": ["rhinoceros"],
        "synset": "rhinoceros.n.01"
    },
    {
        "def": "cut of meat including one or more ribs",
        "frequency": "r",
        "id": 898,
        "name": "rib_(food)",
        "synonyms": ["rib_(food)"],
        "synset": "rib.n.03"
    },
    {
        "def": "a shoulder firearm with a long barrel",
        "frequency": "r",
        "id": 899,
        "name": "rifle",
        "synonyms": ["rifle"],
        "synset": "rifle.n.01"
    },
    {
        "def": "jewelry consisting of a circlet of precious metal (often set with jewels) worn on "
               "the finger",
        "frequency": "f",
        "id": 900,
        "name": "ring",
        "synonyms": ["ring"],
        "synset": "ring.n.08"
    },
    {
        "def": "a boat used on rivers or to ply a river",
        "frequency": "r",
        "id": 901,
        "name": "river_boat",
        "synonyms": ["river_boat"],
        "synset": "river_boat.n.01"
    },
    {
        "def": "(NOT A ROAD) a MAP showing roads (for automobile travel)",
        "frequency": "r",
        "id": 902,
        "name": "road_map",
        "synonyms": ["road_map"],
        "synset": "road_map.n.02"
    },
    {
        "def": "any loose flowing garment",
        "frequency": "c",
        "id": 903,
        "name": "robe",
        "synonyms": ["robe"],
        "synset": "robe.n.01"
    },
    {
        "def": "a chair mounted on rockers",
        "frequency": "c",
        "id": 904,
        "name": "rocking_chair",
        "synonyms": ["rocking_chair"],
        "synset": "rocking_chair.n.01"
    },
    {
        "def": "a shoe with pairs of rollers (small hard wheels) fixed to the sole",
        "frequency": "r",
        "id": 905,
        "name": "roller_skate",
        "synonyms": ["roller_skate"],
        "synset": "roller_skate.n.01"
    },
    {
        "def": "an in-line variant of a roller skate",
        "frequency": "r",
        "id": 906,
        "name": "Rollerblade",
        "synonyms": ["Rollerblade"],
        "synset": "rollerblade.n.01"
    },
    {
        "def": "utensil consisting of a cylinder (usually of wood) with a handle at each end; used "
               "to roll out dough",
        "frequency": "c",
        "id": 907,
        "name": "rolling_pin",
        "synonyms": ["rolling_pin"],
        "synset": "rolling_pin.n.01"
    },
    {
        "def": "carbonated drink containing extracts of roots and herbs",
        "frequency": "r",
        "id": 908,
        "name": "root_beer",
        "synonyms": ["root_beer"],
        "synset": "root_beer.n.01"
    },
    {
        "def": "a device that forwards data packets between computer networks",
        "frequency": "c",
        "id": 909,
        "name": "router_(computer_equipment)",
        "synonyms": ["router_(computer_equipment)"],
        "synset": "router.n.02"
    },
    {
        "def": "a narrow band of elastic rubber used to hold things (such as papers) together",
        "frequency": "f",
        "id": 910,
        "name": "rubber_band",
        "synonyms": ["rubber_band", "elastic_band"],
        "synset": "rubber_band.n.01"
    },
    {
        "def": "a long narrow carpet",
        "frequency": "c",
        "id": 911,
        "name": "runner_(carpet)",
        "synonyms": ["runner_(carpet)"],
        "synset": "runner.n.08"
    },
    {
        "def": "a bag made of paper or plastic for holding customer's purchases",
        "frequency": "f",
        "id": 912,
        "name": "plastic_bag",
        "synonyms": ["plastic_bag", "paper_bag"],
        "synset": "sack.n.01"
    },
    {
        "def": "a seat for the rider of a horse or camel",
        "frequency": "f",
        "id": 913,
        "name": "saddle_(on_an_animal)",
        "synonyms": ["saddle_(on_an_animal)"],
        "synset": "saddle.n.01"
    },
    {
        "def": "stable gear consisting of a blanket placed under the saddle",
        "frequency": "f",
        "id": 914,
        "name": "saddle_blanket",
        "synonyms": ["saddle_blanket", "saddlecloth", "horse_blanket"],
        "synset": "saddle_blanket.n.01"
    },
    {
        "def": "a large bag (or pair of bags) hung over a saddle",
        "frequency": "c",
        "id": 915,
        "name": "saddlebag",
        "synonyms": ["saddlebag"],
        "synset": "saddlebag.n.01"
    },
    {
        "def": "a pin in the form of a clasp; has a guard so the point of the pin will not stick "
               "the user",
        "frequency": "r",
        "id": 916,
        "name": "safety_pin",
        "synonyms": ["safety_pin"],
        "synset": "safety_pin.n.01"
    },
    {
        "def": "a large piece of fabric by means of which wind is used to propel a sailing vessel",
        "frequency": "c",
        "id": 917,
        "name": "sail",
        "synonyms": ["sail"],
        "synset": "sail.n.01"
    },
    {
        "def": "food mixtures either arranged on a plate or tossed and served with a moist "
               "dressing; usually consisting of or including greens",
        "frequency": "c",
        "id": 918,
        "name": "salad",
        "synonyms": ["salad"],
        "synset": "salad.n.01"
    },
    {
        "def": "a plate or bowl for individual servings of salad",
        "frequency": "r",
        "id": 919,
        "name": "salad_plate",
        "synonyms": ["salad_plate", "salad_bowl"],
        "synset": "salad_plate.n.01"
    },
    {
        "def": "highly seasoned fatty sausage of pork and beef usually dried",
        "frequency": "r",
        "id": 920,
        "name": "salami",
        "synonyms": ["salami"],
        "synset": "salami.n.01"
    },
    {
        "def": "any of various large food and game fishes of northern waters",
        "frequency": "r",
        "id": 921,
        "name": "salmon_(fish)",
        "synonyms": ["salmon_(fish)"],
        "synset": "salmon.n.01"
    },
    {
        "def": "flesh of any of various marine or freshwater fish of the family Salmonidae",
        "frequency": "r",
        "id": 922,
        "name": "salmon_(food)",
        "synonyms": ["salmon_(food)"],
        "synset": "salmon.n.03"
    },
    {
        "def": "spicy sauce of tomatoes and onions and chili peppers to accompany Mexican foods",
        "frequency": "r",
        "id": 923,
        "name": "salsa",
        "synonyms": ["salsa"],
        "synset": "salsa.n.01"
    },
    {
        "def": "a shaker with a perforated top for sprinkling salt",
        "frequency": "f",
        "id": 924,
        "name": "saltshaker",
        "synonyms": ["saltshaker"],
        "synset": "saltshaker.n.01"
    },
    {
        "def": "a shoe consisting of a sole fastened by straps to the foot",
        "frequency": "f",
        "id": 925,
        "name": "sandal_(type_of_shoe)",
        "synonyms": ["sandal_(type_of_shoe)"],
        "synset": "sandal.n.01"
    },
    {
        "def": "two (or more) slices of bread with a filling between them",
        "frequency": "f",
        "id": 926,
        "name": "sandwich",
        "synonyms": ["sandwich"],
        "synset": "sandwich.n.01"
    },
    {
        "def": "luggage consisting of a small case with a flat bottom and (usually) a shoulder "
               "strap",
        "frequency": "r",
        "id": 927,
        "name": "satchel",
        "synonyms": ["satchel"],
        "synset": "satchel.n.01"
    },
    {
        "def": "a deep pan with a handle; used for stewing or boiling",
        "frequency": "r",
        "id": 928,
        "name": "saucepan",
        "synonyms": ["saucepan"],
        "synset": "saucepan.n.01"
    },
    {
        "def": "a small shallow dish for holding a cup at the table",
        "frequency": "f",
        "id": 929,
        "name": "saucer",
        "synonyms": ["saucer"],
        "synset": "saucer.n.02"
    },
    {
        "def": "highly seasoned minced meat stuffed in casings",
        "frequency": "f",
        "id": 930,
        "name": "sausage",
        "synonyms": ["sausage"],
        "synset": "sausage.n.01"
    },
    {
        "def": "a framework for holding wood that is being sawed",
        "frequency": "r",
        "id": 931,
        "name": "sawhorse",
        "synonyms": ["sawhorse", "sawbuck"],
        "synset": "sawhorse.n.01"
    },
    {
        "def": "a wind instrument with a `J'-shaped form typically made of brass",
        "frequency": "r",
        "id": 932,
        "name": "saxophone",
        "synonyms": ["saxophone"],
        "synset": "sax.n.02"
    },
    {
        "def": "a measuring instrument for weighing; shows amount of mass",
        "frequency": "f",
        "id": 933,
        "name": "scale_(measuring_instrument)",
        "synonyms": ["scale_(measuring_instrument)"],
        "synset": "scale.n.07"
    },
    {
        "def": "an effigy in the shape of a man to frighten birds away from seeds",
        "frequency": "r",
        "id": 934,
        "name": "scarecrow",
        "synonyms": ["scarecrow", "strawman"],
        "synset": "scarecrow.n.01"
    },
    {
        "def": "a garment worn around the head or neck or shoulders for warmth or decoration",
        "frequency": "f",
        "id": 935,
        "name": "scarf",
        "synonyms": ["scarf"],
        "synset": "scarf.n.01"
    },
    {
        "def": "a bus used to transport children to or from school",
        "frequency": "c",
        "id": 936,
        "name": "school_bus",
        "synonyms": ["school_bus"],
        "synset": "school_bus.n.01"
    },
    {
        "def": "a tool having two crossed pivoting blades with looped handles",
        "frequency": "f",
        "id": 937,
        "name": "scissors",
        "synonyms": ["scissors"],
        "synset": "scissors.n.01"
    },
    {
        "def": "a large board for displaying the score of a contest (and some other information)",
        "frequency": "c",
        "id": 938,
        "name": "scoreboard",
        "synonyms": ["scoreboard"],
        "synset": "scoreboard.n.01"
    },
    {
        "def": "eggs beaten and cooked to a soft firm consistency while stirring",
        "frequency": "c",
        "id": 939,
        "name": "scrambled_eggs",
        "synonyms": ["scrambled_eggs"],
        "synset": "scrambled_eggs.n.01"
    },
    {
        "def": "any of various hand tools for scraping",
        "frequency": "r",
        "id": 940,
        "name": "scraper",
        "synonyms": ["scraper"],
        "synset": "scraper.n.01"
    },
    {
        "def": "a device used for scratching",
        "frequency": "r",
        "id": 941,
        "name": "scratcher",
        "synonyms": ["scratcher"],
        "synset": "scratcher.n.03"
    },
    {
        "def": "a hand tool for driving screws; has a tip that fits into the head of a screw",
        "frequency": "c",
        "id": 942,
        "name": "screwdriver",
        "synonyms": ["screwdriver"],
        "synset": "screwdriver.n.01"
    },
    {
        "def": "a brush with short stiff bristles for heavy cleaning",
        "frequency": "c",
        "id": 943,
        "name": "scrubbing_brush",
        "synonyms": ["scrubbing_brush"],
        "synset": "scrub_brush.n.01"
    },
    {
        "def": "a three-dimensional work of art",
        "frequency": "c",
        "id": 944,
        "name": "sculpture",
        "synonyms": ["sculpture"],
        "synset": "sculpture.n.01"
    },
    {
        "def": "a bird that frequents coastal waters and the open ocean: gulls; pelicans; gannets; "
               "cormorants; albatrosses; petrels; etc.",
        "frequency": "r",
        "id": 945,
        "name": "seabird",
        "synonyms": ["seabird", "seafowl"],
        "synset": "seabird.n.01"
    },
    {
        "def": "small fish with horse-like heads bent sharply downward and curled tails",
        "frequency": "r",
        "id": 946,
        "name": "seahorse",
        "synonyms": ["seahorse"],
        "synset": "seahorse.n.02"
    },
    {
        "def": "an airplane that can land on or take off from water",
        "frequency": "r",
        "id": 947,
        "name": "seaplane",
        "synonyms": ["seaplane", "hydroplane"],
        "synset": "seaplane.n.01"
    },
    {
        "def": "the shell of a marine organism",
        "frequency": "c",
        "id": 948,
        "name": "seashell",
        "synonyms": ["seashell"],
        "synset": "seashell.n.01"
    },
    {
        "def": "young plant or tree grown from a seed",
        "frequency": "r",
        "id": 949,
        "name": "seedling",
        "synonyms": ["seedling"],
        "synset": "seedling.n.01"
    },
    {
        "def": "a dish used for serving food",
        "frequency": "c",
        "id": 950,
        "name": "serving_dish",
        "synonyms": ["serving_dish"],
        "synset": "serving_dish.n.01"
    },
    {
        "def": "a textile machine used as a home appliance for sewing",
        "frequency": "r",
        "id": 951,
        "name": "sewing_machine",
        "synonyms": ["sewing_machine"],
        "synset": "sewing_machine.n.01"
    },
    {
        "def": "a container in which something can be shaken",
        "frequency": "r",
        "id": 952,
        "name": "shaker",
        "synonyms": ["shaker"],
        "synset": "shaker.n.03"
    },
    {
        "def": "cleansing agent consisting of soaps or detergents used for washing the hair",
        "frequency": "c",
        "id": 953,
        "name": "shampoo",
        "synonyms": ["shampoo"],
        "synset": "shampoo.n.01"
    },
    {
        "def": "typically large carnivorous fishes with sharpe teeth",
        "frequency": "r",
        "id": 954,
        "name": "shark",
        "synonyms": ["shark"],
        "synset": "shark.n.01"
    },
    {
        "def": "any implement that is used to make something (an edge or a point) sharper",
        "frequency": "r",
        "id": 955,
        "name": "sharpener",
        "synonyms": ["sharpener"],
        "synset": "sharpener.n.01"
    },
    {
        "def": "a pen with indelible ink that will write on any surface",
        "frequency": "r",
        "id": 956,
        "name": "Sharpie",
        "synonyms": ["Sharpie"],
        "synset": "sharpie.n.03"
    },
    {
        "def": "a razor powered by an electric motor",
        "frequency": "r",
        "id": 957,
        "name": "shaver_(electric)",
        "synonyms": ["shaver_(electric)", "electric_shaver", "electric_razor"],
        "synset": "shaver.n.03"
    },
    {
        "def": "toiletry consisting that forms a rich lather for softening the beard before "
               "shaving",
        "frequency": "c",
        "id": 958,
        "name": "shaving_cream",
        "synonyms": ["shaving_cream", "shaving_soap"],
        "synset": "shaving_cream.n.01"
    },
    {
        "def": "cloak consisting of an oblong piece of cloth used to cover the head and shoulders",
        "frequency": "r",
        "id": 959,
        "name": "shawl",
        "synonyms": ["shawl"],
        "synset": "shawl.n.01"
    },
    {
        "def": "large scissors with strong blades",
        "frequency": "r",
        "id": 960,
        "name": "shears",
        "synonyms": ["shears"],
        "synset": "shears.n.01"
    },
    {
        "def": "woolly usually horned ruminant mammal related to the goat",
        "frequency": "f",
        "id": 961,
        "name": "sheep",
        "synonyms": ["sheep"],
        "synset": "sheep.n.01"
    },
    {
        "def": "any of various usually long-haired breeds of dog reared to herd and guard sheep",
        "frequency": "r",
        "id": 962,
        "name": "shepherd_dog",
        "synonyms": ["shepherd_dog", "sheepdog"],
        "synset": "shepherd_dog.n.01"
    },
    {
        "def": "a frozen dessert made primarily of fruit juice and sugar",
        "frequency": "r",
        "id": 963,
        "name": "sherbert",
        "synonyms": ["sherbert", "sherbet"],
        "synset": "sherbert.n.01"
    },
    {
        "def": "armor carried on the arm to intercept blows",
        "frequency": "r",
        "id": 964,
        "name": "shield",
        "synonyms": ["shield"],
        "synset": "shield.n.02"
    },
    {
        "def": "a garment worn on the upper half of the body",
        "frequency": "f",
        "id": 965,
        "name": "shirt",
        "synonyms": ["shirt"],
        "synset": "shirt.n.01"
    },
    {
        "def": "common footwear covering the foot",
        "frequency": "f",
        "id": 966,
        "name": "shoe",
        "synonyms": ["shoe", "sneaker_(type_of_shoe)", "tennis_shoe"],
        "synset": "shoe.n.01"
    },
    {
        "def": "a bag made of plastic or strong paper (often with handles); used to transport "
               "goods after shopping",
        "frequency": "c",
        "id": 967,
        "name": "shopping_bag",
        "synonyms": ["shopping_bag"],
        "synset": "shopping_bag.n.01"
    },
    {
        "def": "a handcart that holds groceries or other goods while shopping",
        "frequency": "c",
        "id": 968,
        "name": "shopping_cart",
        "synonyms": ["shopping_cart"],
        "synset": "shopping_cart.n.01"
    },
    {
        "def": "trousers that end at or above the knee",
        "frequency": "f",
        "id": 969,
        "name": "short_pants",
        "synonyms": ["short_pants", "shorts_(clothing)", "trunks_(clothing)"],
        "synset": "short_pants.n.01"
    },
    {
        "def": "a small glass adequate to hold a single swallow of whiskey",
        "frequency": "r",
        "id": 970,
        "name": "shot_glass",
        "synonyms": ["shot_glass"],
        "synset": "shot_glass.n.01"
    },
    {
        "def": "a large handbag that can be carried by a strap looped over the shoulder",
        "frequency": "c",
        "id": 971,
        "name": "shoulder_bag",
        "synonyms": ["shoulder_bag"],
        "synset": "shoulder_bag.n.01"
    },
    {
        "def": "a hand tool for lifting loose material such as snow, dirt, etc.",
        "frequency": "c",
        "id": 972,
        "name": "shovel",
        "synonyms": ["shovel"],
        "synset": "shovel.n.01"
    },
    {
        "def": "a plumbing fixture that sprays water over you",
        "frequency": "f",
        "id": 973,
        "name": "shower_head",
        "synonyms": ["shower_head"],
        "synset": "shower.n.01"
    },
    {
        "def": "a curtain that keeps water from splashing out of the shower area",
        "frequency": "f",
        "id": 974,
        "name": "shower_curtain",
        "synonyms": ["shower_curtain"],
        "synset": "shower_curtain.n.01"
    },
    {
        "def": "a device that shreds documents",
        "frequency": "r",
        "id": 975,
        "name": "shredder_(for_paper)",
        "synonyms": ["shredder_(for_paper)"],
        "synset": "shredder.n.01"
    },
    {
        "def": "a strainer for separating lumps from powdered material or grading particles",
        "frequency": "r",
        "id": 976,
        "name": "sieve",
        "synonyms": ["sieve", "screen_(sieve)"],
        "synset": "sieve.n.01"
    },
    {
        "def": "structure displaying a board on which advertisements can be posted",
        "frequency": "f",
        "id": 977,
        "name": "signboard",
        "synonyms": ["signboard"],
        "synset": "signboard.n.01"
    },
    {
        "def": "a cylindrical tower used for storing goods",
        "frequency": "c",
        "id": 978,
        "name": "silo",
        "synonyms": ["silo"],
        "synset": "silo.n.01"
    },
    {
        "def": "plumbing fixture consisting of a water basin fixed to a wall or floor and having a "
               "drainpipe",
        "frequency": "f",
        "id": 979,
        "name": "sink",
        "synonyms": ["sink"],
        "synset": "sink.n.01"
    },
    {
        "def": "a board with wheels that is ridden in a standing or crouching position and "
               "propelled by foot",
        "frequency": "f",
        "id": 980,
        "name": "skateboard",
        "synonyms": ["skateboard"],
        "synset": "skateboard.n.01"
    },
    {
        "def": "a long pin for holding meat in position while it is being roasted",
        "frequency": "c",
        "id": 981,
        "name": "skewer",
        "synonyms": ["skewer"],
        "synset": "skewer.n.01"
    },
    {
        "def": "sports equipment for skiing on snow",
        "frequency": "f",
        "id": 982,
        "name": "ski",
        "synonyms": ["ski"],
        "synset": "ski.n.01"
    },
    {
        "def": "a stiff boot that is fastened to a ski with a ski binding",
        "frequency": "f",
        "id": 983,
        "name": "ski_boot",
        "synonyms": ["ski_boot"],
        "synset": "ski_boot.n.01"
    },
    {
        "def": "a parka to be worn while skiing",
        "frequency": "f",
        "id": 984,
        "name": "ski_parka",
        "synonyms": ["ski_parka", "ski_jacket"],
        "synset": "ski_parka.n.01"
    },
    {
        "def": "a pole with metal points used as an aid in skiing",
        "frequency": "f",
        "id": 985,
        "name": "ski_pole",
        "synonyms": ["ski_pole"],
        "synset": "ski_pole.n.01"
    },
    {
        "def": "a garment hanging from the waist; worn mainly by girls and women",
        "frequency": "f",
        "id": 986,
        "name": "skirt",
        "synonyms": ["skirt"],
        "synset": "skirt.n.02"
    },
    {
        "def": "a vehicle or flat object for transportation over snow by sliding or pulled by "
               "dogs, etc.",
        "frequency": "c",
        "id": 987,
        "name": "sled",
        "synonyms": ["sled", "sledge", "sleigh"],
        "synset": "sled.n.01"
    },
    {
        "def": "large padded bag designed to be slept in outdoors",
        "frequency": "c",
        "id": 988,
        "name": "sleeping_bag",
        "synonyms": ["sleeping_bag"],
        "synset": "sleeping_bag.n.01"
    },
    {
        "def": "bandage to support an injured forearm; slung over the shoulder or neck",
        "frequency": "r",
        "id": 989,
        "name": "sling_(bandage)",
        "synonyms": ["sling_(bandage)", "triangular_bandage"],
        "synset": "sling.n.05"
    },
    {
        "def": "low footwear that can be slipped on and off easily; usually worn indoors",
        "frequency": "c",
        "id": 990,
        "name": "slipper_(footwear)",
        "synonyms": ["slipper_(footwear)", "carpet_slipper_(footwear)"],
        "synset": "slipper.n.01"
    },
    {
        "def": "a thick smooth drink consisting of fresh fruit pureed with ice cream or yoghurt or "
               "milk",
        "frequency": "r",
        "id": 991,
        "name": "smoothie",
        "synonyms": ["smoothie"],
        "synset": "smoothie.n.02"
    },
    {
        "def": "limbless scaly elongate reptile; some are venomous",
        "frequency": "r",
        "id": 992,
        "name": "snake",
        "synonyms": ["snake", "serpent"],
        "synset": "snake.n.01"
    },
    {
        "def": "a board that resembles a broad ski or a small surfboard; used in a standing "
               "position to slide down snow-covered slopes",
        "frequency": "f",
        "id": 993,
        "name": "snowboard",
        "synonyms": ["snowboard"],
        "synset": "snowboard.n.01"
    },
    {
        "def": "a figure of a person made of packed snow",
        "frequency": "c",
        "id": 994,
        "name": "snowman",
        "synonyms": ["snowman"],
        "synset": "snowman.n.01"
    },
    {
        "def": "tracked vehicle for travel on snow having skis in front",
        "frequency": "c",
        "id": 995,
        "name": "snowmobile",
        "synonyms": ["snowmobile"],
        "synset": "snowmobile.n.01"
    },
    {
        "def": "a cleansing agent made from the salts of vegetable or animal fats",
        "frequency": "f",
        "id": 996,
        "name": "soap",
        "synonyms": ["soap"],
        "synset": "soap.n.01"
    },
    {
        "def": "an inflated ball used in playing soccer (called `football' outside of the United "
               "States)",
        "frequency": "f",
        "id": 997,
        "name": "soccer_ball",
        "synonyms": ["soccer_ball"],
        "synset": "soccer_ball.n.01"
    },
    {
        "def": "cloth covering for the foot; worn inside the shoe; reaches to between the ankle "
               "and the knee",
        "frequency": "f",
        "id": 998,
        "name": "sock",
        "synonyms": ["sock"],
        "synset": "sock.n.01"
    },
    {
        "def": "an apparatus for dispensing soda water",
        "frequency": "r",
        "id": 999,
        "name": "soda_fountain",
        "synonyms": ["soda_fountain"],
        "synset": "soda_fountain.n.02"
    },
    {
        "def": "effervescent beverage artificially charged with carbon dioxide",
        "frequency": "r",
        "id": 1000,
        "name": "carbonated_water",
        "synonyms": ["carbonated_water", "club_soda", "seltzer", "sparkling_water"],
        "synset": "soda_water.n.01"
    },
    {
        "def": "an upholstered seat for more than one person",
        "frequency": "f",
        "id": 1001,
        "name": "sofa",
        "synonyms": ["sofa", "couch", "lounge"],
        "synset": "sofa.n.01"
    },
    {
        "def": "ball used in playing softball",
        "frequency": "r",
        "id": 1002,
        "name": "softball",
        "synonyms": ["softball"],
        "synset": "softball.n.01"
    },
    {
        "def": "electrical device consisting of a large array of connected solar cells",
        "frequency": "c",
        "id": 1003,
        "name": "solar_array",
        "synonyms": ["solar_array", "solar_battery", "solar_panel"],
        "synset": "solar_array.n.01"
    },
    {
        "def": "a straw hat with a tall crown and broad brim; worn in American southwest and in "
               "Mexico",
        "frequency": "r",
        "id": 1004,
        "name": "sombrero",
        "synonyms": ["sombrero"],
        "synset": "sombrero.n.02"
    },
    {
        "def": "liquid food especially of meat or fish or vegetable stock often containing pieces "
               "of solid food",
        "frequency": "c",
        "id": 1005,
        "name": "soup",
        "synonyms": ["soup"],
        "synset": "soup.n.01"
    },
    {
        "def": "a bowl for serving soup",
        "frequency": "r",
        "id": 1006,
        "name": "soup_bowl",
        "synonyms": ["soup_bowl"],
        "synset": "soup_bowl.n.01"
    },
    {
        "def": "a spoon with a rounded bowl for eating soup",
        "frequency": "c",
        "id": 1007,
        "name": "soupspoon",
        "synonyms": ["soupspoon"],
        "synset": "soupspoon.n.01"
    },
    {
        "def": "soured light cream",
        "frequency": "c",
        "id": 1008,
        "name": "sour_cream",
        "synonyms": ["sour_cream", "soured_cream"],
        "synset": "sour_cream.n.01"
    },
    {
        "def": "a milk substitute containing soybean flour and water; used in some infant formulas "
               "and in making tofu",
        "frequency": "r",
        "id": 1009,
        "name": "soya_milk",
        "synonyms": ["soya_milk", "soybean_milk", "soymilk"],
        "synset": "soya_milk.n.01"
    },
    {
        "def": "a reusable spacecraft with wings for a controlled descent through the Earth's "
               "atmosphere",
        "frequency": "r",
        "id": 1010,
        "name": "space_shuttle",
        "synonyms": ["space_shuttle"],
        "synset": "space_shuttle.n.01"
    },
    {
        "def": "a firework that burns slowly and throws out a shower of sparks",
        "frequency": "r",
        "id": 1011,
        "name": "sparkler_(fireworks)",
        "synonyms": ["sparkler_(fireworks)"],
        "synset": "sparkler.n.02"
    },
    {
        "def": "a hand tool with a thin flexible blade used to mix or spread soft substances",
        "frequency": "f",
        "id": 1012,
        "name": "spatula",
        "synonyms": ["spatula"],
        "synset": "spatula.n.02"
    },
    {
        "def": "a long pointed rod used as a tool or weapon",
        "frequency": "r",
        "id": 1013,
        "name": "spear",
        "synonyms": ["spear", "lance"],
        "synset": "spear.n.01"
    },
    {
        "def": "optical instrument consisting of a frame that holds a pair of lenses for "
               "correcting defective vision",
        "frequency": "f",
        "id": 1014,
        "name": "spectacles",
        "synonyms": ["spectacles", "specs", "eyeglasses", "glasses"],
        "synset": "spectacles.n.01"
    },
    {
        "def": "a rack for displaying containers filled with spices",
        "frequency": "c",
        "id": 1015,
        "name": "spice_rack",
        "synonyms": ["spice_rack"],
        "synset": "spice_rack.n.01"
    },
    {
        "def": "predatory arachnid with eight legs, two poison fangs, two feelers, and usually two "
               "silk-spinning organs at the back end of the body",
        "frequency": "r",
        "id": 1016,
        "name": "spider",
        "synonyms": ["spider"],
        "synset": "spider.n.01"
    },
    {
        "def": "a porous mass usable to absorb water typically used for cleaning",
        "frequency": "c",
        "id": 1017,
        "name": "sponge",
        "synonyms": ["sponge"],
        "synset": "sponge.n.01"
    },
    {
        "def": "a piece of cutlery with a shallow bowl-shaped container and a handle",
        "frequency": "f",
        "id": 1018,
        "name": "spoon",
        "synonyms": ["spoon"],
        "synset": "spoon.n.01"
    },
    {
        "def": "attire worn for sport or for casual wear",
        "frequency": "c",
        "id": 1019,
        "name": "sportswear",
        "synonyms": ["sportswear", "athletic_wear", "activewear"],
        "synset": "sportswear.n.01"
    },
    {
        "def": "a lamp that produces a strong beam of light to illuminate a restricted area; used "
               "to focus attention of a stage performer",
        "frequency": "c",
        "id": 1020,
        "name": "spotlight",
        "synonyms": ["spotlight"],
        "synset": "spotlight.n.02"
    },
    {
        "def": "a kind of arboreal rodent having a long bushy tail",
        "frequency": "r",
        "id": 1021,
        "name": "squirrel",
        "synonyms": ["squirrel"],
        "synset": "squirrel.n.01"
    },
    {
        "def": "a machine that inserts staples into sheets of paper in order to fasten them "
               "together",
        "frequency": "c",
        "id": 1022,
        "name": "stapler_(stapling_machine)",
        "synonyms": ["stapler_(stapling_machine)"],
        "synset": "stapler.n.01"
    },
    {
        "def": "echinoderms characterized by five arms extending from a central disk",
        "frequency": "r",
        "id": 1023,
        "name": "starfish",
        "synonyms": ["starfish", "sea_star"],
        "synset": "starfish.n.01"
    },
    {
        "def": "a sculpture representing a human or animal",
        "frequency": "f",
        "id": 1024,
        "name": "statue_(sculpture)",
        "synonyms": ["statue_(sculpture)"],
        "synset": "statue.n.01"
    },
    {
        "def": "a slice of meat cut from the fleshy part of an animal or large fish",
        "frequency": "c",
        "id": 1025,
        "name": "steak_(food)",
        "synonyms": ["steak_(food)"],
        "synset": "steak.n.01"
    },
    {
        "def": "a sharp table knife used in eating steak",
        "frequency": "r",
        "id": 1026,
        "name": "steak_knife",
        "synonyms": ["steak_knife"],
        "synset": "steak_knife.n.01"
    },
    {
        "def": "a cooking utensil that can be used to cook food by steaming it",
        "frequency": "r",
        "id": 1027,
        "name": "steamer_(kitchen_appliance)",
        "synonyms": ["steamer_(kitchen_appliance)"],
        "synset": "steamer.n.02"
    },
    {
        "def": "a handwheel that is used for steering",
        "frequency": "f",
        "id": 1028,
        "name": "steering_wheel",
        "synonyms": ["steering_wheel"],
        "synset": "steering_wheel.n.01"
    },
    {
        "def": "a sheet of material (metal, plastic, etc.) that has been perforated with a "
               "pattern; ink or paint can pass through the perforations to create the printed "
               "pattern on the surface below",
        "frequency": "r",
        "id": 1029,
        "name": "stencil",
        "synonyms": ["stencil"],
        "synset": "stencil.n.01"
    },
    {
        "def": "a folding portable ladder hinged at the top",
        "frequency": "r",
        "id": 1030,
        "name": "stepladder",
        "synonyms": ["stepladder"],
        "synset": "step_ladder.n.01"
    },
    {
        "def": "a stool that has one or two steps that fold under the seat",
        "frequency": "c",
        "id": 1031,
        "name": "step_stool",
        "synonyms": ["step_stool"],
        "synset": "step_stool.n.01"
    },
    {
        "def": "electronic device for playing audio",
        "frequency": "c",
        "id": 1032,
        "name": "stereo_(sound_system)",
        "synonyms": ["stereo_(sound_system)"],
        "synset": "stereo.n.01"
    },
    {
        "def": "food prepared by stewing especially meat or fish with vegetables",
        "frequency": "r",
        "id": 1033,
        "name": "stew",
        "synonyms": ["stew"],
        "synset": "stew.n.02"
    },
    {
        "def": "an implement used for stirring",
        "frequency": "r",
        "id": 1034,
        "name": "stirrer",
        "synonyms": ["stirrer"],
        "synset": "stirrer.n.02"
    },
    {
        "def": "support consisting of metal loops into which rider's feet go",
        "frequency": "f",
        "id": 1035,
        "name": "stirrup",
        "synonyms": ["stirrup"],
        "synset": "stirrup.n.01"
    },
    {
        "def": "close-fitting hosiery to cover the foot and leg; come in matched pairs",
        "frequency": "c",
        "id": 1036,
        "name": "stockings_(leg_wear)",
        "synonyms": ["stockings_(leg_wear)"],
        "synset": "stocking.n.01"
    },
    {
        "def": "a simple seat without a back or arms",
        "frequency": "f",
        "id": 1037,
        "name": "stool",
        "synonyms": ["stool"],
        "synset": "stool.n.01"
    },
    {
        "def": "a traffic sign to notify drivers that they must come to a complete stop",
        "frequency": "f",
        "id": 1038,
        "name": "stop_sign",
        "synonyms": ["stop_sign"],
        "synset": "stop_sign.n.01"
    },
    {
        "def": "a red light on the rear of a motor vehicle that signals when the brakes are "
               "applied",
        "frequency": "f",
        "id": 1039,
        "name": "brake_light",
        "synonyms": ["brake_light"],
        "synset": "stoplight.n.01"
    },
    {
        "def": "a kitchen appliance used for cooking food",
        "frequency": "f",
        "id": 1040,
        "name": "stove",
        "synonyms": ["stove", "kitchen_stove", "range_(kitchen_appliance)", "kitchen_range",
                     "cooking_stove"],
        "synset": "stove.n.01"
    },
    {
        "def": "a filter to retain larger pieces while smaller pieces and liquids pass through",
        "frequency": "c",
        "id": 1041,
        "name": "strainer",
        "synonyms": ["strainer"],
        "synset": "strainer.n.01"
    },
    {
        "def": "an elongated strip of material for binding things together or holding",
        "frequency": "f",
        "id": 1042,
        "name": "strap",
        "synonyms": ["strap"],
        "synset": "strap.n.01"
    },
    {
        "def": "a thin paper or plastic tube used to suck liquids into the mouth",
        "frequency": "f",
        "id": 1043,
        "name": "straw_(for_drinking)",
        "synonyms": ["straw_(for_drinking)", "drinking_straw"],
        "synset": "straw.n.04"
    },
    {
        "def": "sweet fleshy red fruit",
        "frequency": "f",
        "id": 1044,
        "name": "strawberry",
        "synonyms": ["strawberry"],
        "synset": "strawberry.n.01"
    },
    {
        "def": "a sign visible from the street",
        "frequency": "f",
        "id": 1045,
        "name": "street_sign",
        "synonyms": ["street_sign"],
        "synset": "street_sign.n.01"
    },
    {
        "def": "a lamp supported on a lamppost; for illuminating a street",
        "frequency": "f",
        "id": 1046,
        "name": "streetlight",
        "synonyms": ["streetlight", "street_lamp"],
        "synset": "streetlight.n.01"
    },
    {
        "def": "cheese formed in long strings twisted together",
        "frequency": "r",
        "id": 1047,
        "name": "string_cheese",
        "synonyms": ["string_cheese"],
        "synset": "string_cheese.n.01"
    },
    {
        "def": "a pointed tool for writing or drawing or engraving",
        "frequency": "r",
        "id": 1048,
        "name": "stylus",
        "synonyms": ["stylus"],
        "synset": "stylus.n.02"
    },
    {
        "def": "a loudspeaker that is designed to reproduce very low bass frequencies",
        "frequency": "r",
        "id": 1049,
        "name": "subwoofer",
        "synonyms": ["subwoofer"],
        "synset": "subwoofer.n.01"
    },
    {
        "def": "a dish in which sugar is served",
        "frequency": "r",
        "id": 1050,
        "name": "sugar_bowl",
        "synonyms": ["sugar_bowl"],
        "synset": "sugar_bowl.n.01"
    },
    {
        "def": "juicy canes whose sap is a source of molasses and commercial sugar; fresh canes "
               "are sometimes chewed for the juice",
        "frequency": "r",
        "id": 1051,
        "name": "sugarcane_(plant)",
        "synonyms": ["sugarcane_(plant)"],
        "synset": "sugarcane.n.01"
    },
    {
        "def": "a set of garments (usually including a jacket and trousers or skirt) for outerwear "
               "all of the same fabric and color",
        "frequency": "c",
        "id": 1052,
        "name": "suit_(clothing)",
        "synonyms": ["suit_(clothing)"],
        "synset": "suit.n.01"
    },
    {
        "def": "any plant of the genus Helianthus having large flower heads with dark disk florets "
               "and showy yellow rays",
        "frequency": "c",
        "id": 1053,
        "name": "sunflower",
        "synonyms": ["sunflower"],
        "synset": "sunflower.n.01"
    },
    {
        "def": "spectacles that are darkened or polarized to protect the eyes from the glare of "
               "the sun",
        "frequency": "f",
        "id": 1054,
        "name": "sunglasses",
        "synonyms": ["sunglasses"],
        "synset": "sunglasses.n.01"
    },
    {
        "def": "a hat with a broad brim that protects the face from direct exposure to the sun",
        "frequency": "c",
        "id": 1055,
        "name": "sunhat",
        "synonyms": ["sunhat"],
        "synset": "sunhat.n.01"
    },
    {
        "def": "a cream spread on the skin; contains a chemical to filter out ultraviolet light "
               "and so protect from sunburn",
        "frequency": "r",
        "id": 1056,
        "name": "sunscreen",
        "synonyms": ["sunscreen", "sunblock"],
        "synset": "sunscreen.n.01"
    },
    {
        "def": "a narrow buoyant board for riding surf",
        "frequency": "f",
        "id": 1057,
        "name": "surfboard",
        "synonyms": ["surfboard"],
        "synset": "surfboard.n.01"
    },
    {
        "def": "rice (with raw fish) wrapped in seaweed",
        "frequency": "c",
        "id": 1058,
        "name": "sushi",
        "synonyms": ["sushi"],
        "synset": "sushi.n.01"
    },
    {
        "def": "cleaning implement consisting of absorbent material fastened to a handle; for "
               "cleaning floors",
        "frequency": "c",
        "id": 1059,
        "name": "mop",
        "synonyms": ["mop"],
        "synset": "swab.n.02"
    },
    {
        "def": "loose-fitting trousers with elastic cuffs; worn by athletes",
        "frequency": "c",
        "id": 1060,
        "name": "sweat_pants",
        "synonyms": ["sweat_pants"],
        "synset": "sweat_pants.n.01"
    },
    {
        "def": "a band of material tied around the forehead or wrist to absorb sweat",
        "frequency": "c",
        "id": 1061,
        "name": "sweatband",
        "synonyms": ["sweatband"],
        "synset": "sweatband.n.02"
    },
    {
        "def": "a crocheted or knitted garment covering the upper part of the body",
        "frequency": "f",
        "id": 1062,
        "name": "sweater",
        "synonyms": ["sweater"],
        "synset": "sweater.n.01"
    },
    {
        "def": "cotton knit pullover with long sleeves worn during athletic activity",
        "frequency": "f",
        "id": 1063,
        "name": "sweatshirt",
        "synonyms": ["sweatshirt"],
        "synset": "sweatshirt.n.01"
    },
    {
        "def": "the edible tuberous root of the sweet potato vine",
        "frequency": "c",
        "id": 1064,
        "name": "sweet_potato",
        "synonyms": ["sweet_potato"],
        "synset": "sweet_potato.n.02"
    },
    {
        "def": "garment worn for swimming",
        "frequency": "f",
        "id": 1065,
        "name": "swimsuit",
        "synonyms": ["swimsuit", "swimwear", "bathing_suit", "swimming_costume",
                     "bathing_costume", "swimming_trunks", "bathing_trunks"],
        "synset": "swimsuit.n.01"
    },
    {
        "def": "a cutting or thrusting weapon that has a long metal blade",
        "frequency": "c",
        "id": 1066,
        "name": "sword",
        "synonyms": ["sword"],
        "synset": "sword.n.01"
    },
    {
        "def": "a medical instrument used to inject or withdraw fluids",
        "frequency": "r",
        "id": 1067,
        "name": "syringe",
        "synonyms": ["syringe"],
        "synset": "syringe.n.01"
    },
    {
        "def": "very spicy sauce (trade name Tabasco) made from fully-aged red peppers",
        "frequency": "r",
        "id": 1068,
        "name": "Tabasco_sauce",
        "synonyms": ["Tabasco_sauce"],
        "synset": "tabasco.n.02"
    },
    {
        "def": "a table used for playing table tennis",
        "frequency": "r",
        "id": 1069,
        "name": "table-tennis_table",
        "synonyms": ["table-tennis_table", "ping-pong_table"],
        "synset": "table-tennis_table.n.01"
    },
    {
        "def": "a piece of furniture having a smooth flat top that is usually supported by one or "
               "more vertical legs",
        "frequency": "f",
        "id": 1070,
        "name": "table",
        "synonyms": ["table"],
        "synset": "table.n.02"
    },
    {
        "def": "a lamp that sits on a table",
        "frequency": "c",
        "id": 1071,
        "name": "table_lamp",
        "synonyms": ["table_lamp"],
        "synset": "table_lamp.n.01"
    },
    {
        "def": "a covering spread over a dining table",
        "frequency": "f",
        "id": 1072,
        "name": "tablecloth",
        "synonyms": ["tablecloth"],
        "synset": "tablecloth.n.01"
    },
    {
        "def": "measuring instrument for indicating speed of rotation",
        "frequency": "r",
        "id": 1073,
        "name": "tachometer",
        "synonyms": ["tachometer"],
        "synset": "tachometer.n.01"
    },
    {
        "def": "a small tortilla cupped around a filling",
        "frequency": "r",
        "id": 1074,
        "name": "taco",
        "synonyms": ["taco"],
        "synset": "taco.n.02"
    },
    {
        "def": "a label associated with something for the purpose of identification or information",
        "frequency": "f",
        "id": 1075,
        "name": "tag",
        "synonyms": ["tag"],
        "synset": "tag.n.02"
    },
    {
        "def": "lamp (usually red) mounted at the rear of a motor vehicle",
        "frequency": "f",
        "id": 1076,
        "name": "taillight",
        "synonyms": ["taillight", "rear_light"],
        "synset": "taillight.n.01"
    },
    {
        "def": "a shallow drum with a single drumhead and with metallic disks in the sides",
        "frequency": "r",
        "id": 1077,
        "name": "tambourine",
        "synonyms": ["tambourine"],
        "synset": "tambourine.n.01"
    },
    {
        "def": "an enclosed armored military vehicle; has a cannon and moves on caterpillar treads",
        "frequency": "r",
        "id": 1078,
        "name": "army_tank",
        "synonyms": ["army_tank", "armored_combat_vehicle", "armoured_combat_vehicle"],
        "synset": "tank.n.01"
    },
    {
        "def": "a large (usually metallic) vessel for holding gases or liquids",
        "frequency": "c",
        "id": 1079,
        "name": "tank_(storage_vessel)",
        "synonyms": ["tank_(storage_vessel)", "storage_tank"],
        "synset": "tank.n.02"
    },
    {
        "def": "a tight-fitting sleeveless shirt with wide shoulder straps and low neck and no "
               "front opening",
        "frequency": "f",
        "id": 1080,
        "name": "tank_top_(clothing)",
        "synonyms": ["tank_top_(clothing)"],
        "synset": "tank_top.n.01"
    },
    {
        "def": "a long thin piece of cloth or paper as used for binding or fastening",
        "frequency": "c",
        "id": 1081,
        "name": "tape_(sticky_cloth_or_paper)",
        "synonyms": ["tape_(sticky_cloth_or_paper)"],
        "synset": "tape.n.01"
    },
    {
        "def": "measuring instrument consisting of a narrow strip (cloth or metal) marked in "
               "inches or centimeters and used for measuring lengths",
        "frequency": "c",
        "id": 1082,
        "name": "tape_measure",
        "synonyms": ["tape_measure", "measuring_tape"],
        "synset": "tape.n.04"
    },
    {
        "def": "a heavy textile with a woven design; used for curtains and upholstery",
        "frequency": "c",
        "id": 1083,
        "name": "tapestry",
        "synonyms": ["tapestry"],
        "synset": "tapestry.n.02"
    },
    {
        "def": "waterproofed canvas",
        "frequency": "f",
        "id": 1084,
        "name": "tarp",
        "synonyms": ["tarp"],
        "synset": "tarpaulin.n.01"
    },
    {
        "def": "a cloth having a crisscross design",
        "frequency": "c",
        "id": 1085,
        "name": "tartan",
        "synonyms": ["tartan", "plaid"],
        "synset": "tartan.n.01"
    },
    {
        "def": "adornment consisting of a bunch of cords fastened at one end",
        "frequency": "c",
        "id": 1086,
        "name": "tassel",
        "synonyms": ["tassel"],
        "synset": "tassel.n.01"
    },
    {
        "def": "a measured amount of tea in a bag for an individual serving of tea",
        "frequency": "r",
        "id": 1087,
        "name": "tea_bag",
        "synonyms": ["tea_bag"],
        "synset": "tea_bag.n.01"
    },
    {
        "def": "a cup from which tea is drunk",
        "frequency": "c",
        "id": 1088,
        "name": "teacup",
        "synonyms": ["teacup"],
        "synset": "teacup.n.02"
    },
    {
        "def": "kettle for boiling water to make tea",
        "frequency": "c",
        "id": 1089,
        "name": "teakettle",
        "synonyms": ["teakettle"],
        "synset": "teakettle.n.01"
    },
    {
        "def": "pot for brewing tea; usually has a spout and handle",
        "frequency": "c",
        "id": 1090,
        "name": "teapot",
        "synonyms": ["teapot"],
        "synset": "teapot.n.01"
    },
    {
        "def": "plaything consisting of a child's toy bear (usually plush and stuffed with soft "
               "materials)",
        "frequency": "f",
        "id": 1091,
        "name": "teddy_bear",
        "synonyms": ["teddy_bear"],
        "synset": "teddy.n.01"
    },
    {
        "def": "electronic device for communicating by voice over long distances",
        "frequency": "f",
        "id": 1092,
        "name": "telephone",
        "synonyms": ["telephone", "phone", "telephone_set"],
        "synset": "telephone.n.01"
    },
    {
        "def": "booth for using a telephone",
        "frequency": "c",
        "id": 1093,
        "name": "telephone_booth",
        "synonyms": ["telephone_booth", "phone_booth", "call_box", "telephone_box",
                     "telephone_kiosk"],
        "synset": "telephone_booth.n.01"
    },
    {
        "def": "tall pole supporting telephone wires",
        "frequency": "f",
        "id": 1094,
        "name": "telephone_pole",
        "synonyms": ["telephone_pole", "telegraph_pole", "telegraph_post"],
        "synset": "telephone_pole.n.01"
    },
    {
        "def": "a camera lens that magnifies the image",
        "frequency": "r",
        "id": 1095,
        "name": "telephoto_lens",
        "synonyms": ["telephoto_lens", "zoom_lens"],
        "synset": "telephoto_lens.n.01"
    },
    {
        "def": "television equipment for capturing and recording video",
        "frequency": "c",
        "id": 1096,
        "name": "television_camera",
        "synonyms": ["television_camera", "tv_camera"],
        "synset": "television_camera.n.01"
    },
    {
        "def": "an electronic device that receives television signals and displays them on a "
               "screen",
        "frequency": "f",
        "id": 1097,
        "name": "television_set",
        "synonyms": ["television_set", "tv", "tv_set"],
        "synset": "television_receiver.n.01"
    },
    {
        "def": "ball about the size of a fist used in playing tennis",
        "frequency": "f",
        "id": 1098,
        "name": "tennis_ball",
        "synonyms": ["tennis_ball"],
        "synset": "tennis_ball.n.01"
    },
    {
        "def": "a racket used to play tennis",
        "frequency": "f",
        "id": 1099,
        "name": "tennis_racket",
        "synonyms": ["tennis_racket"],
        "synset": "tennis_racket.n.01"
    },
    {
        "def": "Mexican liquor made from fermented juices of an agave plant",
        "frequency": "r",
        "id": 1100,
        "name": "tequila",
        "synonyms": ["tequila"],
        "synset": "tequila.n.01"
    },
    {
        "def": "measuring instrument for measuring temperature",
        "frequency": "c",
        "id": 1101,
        "name": "thermometer",
        "synonyms": ["thermometer"],
        "synset": "thermometer.n.01"
    },
    {
        "def": "vacuum flask that preserves temperature of hot or cold drinks",
        "frequency": "c",
        "id": 1102,
        "name": "thermos_bottle",
        "synonyms": ["thermos_bottle"],
        "synset": "thermos.n.01"
    },
    {
        "def": "a regulator for automatically regulating temperature by starting or stopping the "
               "supply of heat",
        "frequency": "c",
        "id": 1103,
        "name": "thermostat",
        "synonyms": ["thermostat"],
        "synset": "thermostat.n.01"
    },
    {
        "def": "a small metal cap to protect the finger while sewing; can be used as a small "
               "container",
        "frequency": "r",
        "id": 1104,
        "name": "thimble",
        "synonyms": ["thimble"],
        "synset": "thimble.n.02"
    },
    {
        "def": "a fine cord of twisted fibers (of cotton or silk or wool or nylon etc.) used in "
               "sewing and weaving",
        "frequency": "c",
        "id": 1105,
        "name": "thread",
        "synonyms": ["thread", "yarn"],
        "synset": "thread.n.01"
    },
    {
        "def": "a tack for attaching papers to a bulletin board or drawing board",
        "frequency": "c",
        "id": 1106,
        "name": "thumbtack",
        "synonyms": ["thumbtack", "drawing_pin", "pushpin"],
        "synset": "thumbtack.n.01"
    },
    {
        "def": "a jeweled headdress worn by women on formal occasions",
        "frequency": "c",
        "id": 1107,
        "name": "tiara",
        "synonyms": ["tiara"],
        "synset": "tiara.n.01"
    },
    {
        "def": "large feline of forests in most of Asia having a tawny coat with black stripes",
        "frequency": "c",
        "id": 1108,
        "name": "tiger",
        "synonyms": ["tiger"],
        "synset": "tiger.n.02"
    },
    {
        "def": "skintight knit hose covering the body from the waist to the feet worn by acrobats "
               "and dancers and as stockings by women and girls",
        "frequency": "c",
        "id": 1109,
        "name": "tights_(clothing)",
        "synonyms": ["tights_(clothing)", "leotards"],
        "synset": "tights.n.01"
    },
    {
        "def": "a timepiece that measures a time interval and signals its end",
        "frequency": "c",
        "id": 1110,
        "name": "timer",
        "synonyms": ["timer", "stopwatch"],
        "synset": "timer.n.01"
    },
    {
        "def": "foil made of tin or an alloy of tin and lead",
        "frequency": "f",
        "id": 1111,
        "name": "tinfoil",
        "synonyms": ["tinfoil"],
        "synset": "tinfoil.n.01"
    },
    {
        "def": "a showy decoration that is basically valueless",
        "frequency": "r",
        "id": 1112,
        "name": "tinsel",
        "synonyms": ["tinsel"],
        "synset": "tinsel.n.01"
    },
    {
        "def": "a soft thin (usually translucent) paper",
        "frequency": "f",
        "id": 1113,
        "name": "tissue_paper",
        "synonyms": ["tissue_paper"],
        "synset": "tissue.n.02"
    },
    {
        "def": "slice of bread that has been toasted",
        "frequency": "c",
        "id": 1114,
        "name": "toast_(food)",
        "synonyms": ["toast_(food)"],
        "synset": "toast.n.01"
    },
    {
        "def": "a kitchen appliance (usually electric) for toasting bread",
        "frequency": "f",
        "id": 1115,
        "name": "toaster",
        "synonyms": ["toaster"],
        "synset": "toaster.n.02"
    },
    {
        "def": "kitchen appliance consisting of a small electric oven for toasting or warming food",
        "frequency": "c",
        "id": 1116,
        "name": "toaster_oven",
        "synonyms": ["toaster_oven"],
        "synset": "toaster_oven.n.01"
    },
    {
        "def": "a plumbing fixture for defecation and urination",
        "frequency": "f",
        "id": 1117,
        "name": "toilet",
        "synonyms": ["toilet"],
        "synset": "toilet.n.02"
    },
    {
        "def": "a soft thin absorbent paper for use in toilets",
        "frequency": "f",
        "id": 1118,
        "name": "toilet_tissue",
        "synonyms": ["toilet_tissue", "toilet_paper", "bathroom_tissue"],
        "synset": "toilet_tissue.n.01"
    },
    {
        "def": "mildly acid red or yellow pulpy fruit eaten as a vegetable",
        "frequency": "f",
        "id": 1119,
        "name": "tomato",
        "synonyms": ["tomato"],
        "synset": "tomato.n.01"
    },
    {
        "def": "any of various devices for taking hold of objects; usually have two hinged legs "
               "with handles above and pointed hooks below",
        "frequency": "c",
        "id": 1120,
        "name": "tongs",
        "synonyms": ["tongs"],
        "synset": "tongs.n.01"
    },
    {
        "def": "a box or chest or cabinet for holding hand tools",
        "frequency": "c",
        "id": 1121,
        "name": "toolbox",
        "synonyms": ["toolbox"],
        "synset": "toolbox.n.01"
    },
    {
        "def": "small brush; has long handle; used to clean teeth",
        "frequency": "f",
        "id": 1122,
        "name": "toothbrush",
        "synonyms": ["toothbrush"],
        "synset": "toothbrush.n.01"
    },
    {
        "def": "a dentifrice in the form of a paste",
        "frequency": "f",
        "id": 1123,
        "name": "toothpaste",
        "synonyms": ["toothpaste"],
        "synset": "toothpaste.n.01"
    },
    {
        "def": "pick consisting of a small strip of wood or plastic; used to pick food from "
               "between the teeth",
        "frequency": "c",
        "id": 1124,
        "name": "toothpick",
        "synonyms": ["toothpick"],
        "synset": "toothpick.n.01"
    },
    {
        "def": "covering for a hole (especially a hole in the top of a container)",
        "frequency": "c",
        "id": 1125,
        "name": "cover",
        "synonyms": ["cover"],
        "synset": "top.n.09"
    },
    {
        "def": "thin unleavened pancake made from cornmeal or wheat flour",
        "frequency": "c",
        "id": 1126,
        "name": "tortilla",
        "synonyms": ["tortilla"],
        "synset": "tortilla.n.01"
    },
    {
        "def": "a truck equipped to hoist and pull wrecked cars (or to remove cars from no-parking "
               "zones)",
        "frequency": "c",
        "id": 1127,
        "name": "tow_truck",
        "synonyms": ["tow_truck"],
        "synset": "tow_truck.n.01"
    },
    {
        "def": "a rectangular piece of absorbent cloth (or paper) for drying or wiping",
        "frequency": "f",
        "id": 1128,
        "name": "towel",
        "synonyms": ["towel"],
        "synset": "towel.n.01"
    },
    {
        "def": "a rack consisting of one or more bars on which towels can be hung",
        "frequency": "f",
        "id": 1129,
        "name": "towel_rack",
        "synonyms": ["towel_rack", "towel_rail", "towel_bar"],
        "synset": "towel_rack.n.01"
    },
    {
        "def": "a device regarded as providing amusement",
        "frequency": "f",
        "id": 1130,
        "name": "toy",
        "synonyms": ["toy"],
        "synset": "toy.n.03"
    },
    {
        "def": "a wheeled vehicle with large wheels; used in farming and other applications",
        "frequency": "c",
        "id": 1131,
        "name": "tractor_(farm_equipment)",
        "synonyms": ["tractor_(farm_equipment)"],
        "synset": "tractor.n.01"
    },
    {
        "def": "a device to control vehicle traffic often consisting of three or more lights",
        "frequency": "f",
        "id": 1132,
        "name": "traffic_light",
        "synonyms": ["traffic_light"],
        "synset": "traffic_light.n.01"
    },
    {
        "def": "a lightweight motorcycle equipped with rugged tires and suspension for off-road "
               "use",
        "frequency": "r",
        "id": 1133,
        "name": "dirt_bike",
        "synonyms": ["dirt_bike"],
        "synset": "trail_bike.n.01"
    },
    {
        "def": "a truck consisting of a tractor and trailer together",
        "frequency": "c",
        "id": 1134,
        "name": "trailer_truck",
        "synonyms": ["trailer_truck", "tractor_trailer", "trucking_rig", "articulated_lorry",
                     "semi_truck"],
        "synset": "trailer_truck.n.01"
    },
    {
        "def": "public or private transport provided by a line of railway cars coupled together "
               "and drawn by a locomotive",
        "frequency": "f",
        "id": 1135,
        "name": "train_(railroad_vehicle)",
        "synonyms": ["train_(railroad_vehicle)", "railroad_train"],
        "synset": "train.n.01"
    },
    {
        "def": "gymnastic apparatus consisting of a strong canvas sheet attached with springs to a "
               "metal frame",
        "frequency": "r",
        "id": 1136,
        "name": "trampoline",
        "synonyms": ["trampoline"],
        "synset": "trampoline.n.01"
    },
    {
        "def": "an open receptacle for holding or displaying or serving articles or food",
        "frequency": "f",
        "id": 1137,
        "name": "tray",
        "synonyms": ["tray"],
        "synset": "tray.n.01"
    },
    {
        "def": "(NOT A TREE) a PLAYHOUSE built in the branches of a tree",
        "frequency": "r",
        "id": 1138,
        "name": "tree_house",
        "synonyms": ["tree_house"],
        "synset": "tree_house.n.01"
    },
    {
        "def": "a military style raincoat; belted with deep pockets",
        "frequency": "r",
        "id": 1139,
        "name": "trench_coat",
        "synonyms": ["trench_coat"],
        "synset": "trench_coat.n.01"
    },
    {
        "def": "a percussion instrument consisting of a metal bar bent in the shape of an open "
               "triangle",
        "frequency": "r",
        "id": 1140,
        "name": "triangle_(musical_instrument)",
        "synonyms": ["triangle_(musical_instrument)"],
        "synset": "triangle.n.05"
    },
    {
        "def": "a vehicle with three wheels that is moved by foot pedals",
        "frequency": "r",
        "id": 1141,
        "name": "tricycle",
        "synonyms": ["tricycle"],
        "synset": "tricycle.n.01"
    },
    {
        "def": "a three-legged rack used for support",
        "frequency": "c",
        "id": 1142,
        "name": "tripod",
        "synonyms": ["tripod"],
        "synset": "tripod.n.01"
    },
    {
        "def": "a garment extending from the waist to the knee or ankle, covering each leg "
               "separately",
        "frequency": "f",
        "id": 1143,
        "name": "trousers",
        "synonyms": ["trousers", "pants_(clothing)"],
        "synset": "trouser.n.01"
    },
    {
        "def": "an automotive vehicle suitable for hauling",
        "frequency": "f",
        "id": 1144,
        "name": "truck",
        "synonyms": ["truck"],
        "synset": "truck.n.01"
    },
    {
        "def": "creamy chocolate candy",
        "frequency": "r",
        "id": 1145,
        "name": "truffle_(chocolate)",
        "synonyms": ["truffle_(chocolate)", "chocolate_truffle"],
        "synset": "truffle.n.03"
    },
    {
        "def": "luggage consisting of a large strong case used when traveling or for storage",
        "frequency": "c",
        "id": 1146,
        "name": "trunk",
        "synonyms": ["trunk"],
        "synset": "trunk.n.02"
    },
    {
        "def": "a large open vessel for holding or storing liquids",
        "frequency": "r",
        "id": 1147,
        "name": "vat",
        "synonyms": ["vat"],
        "synset": "tub.n.02"
    },
    {
        "def": "a traditional headdress consisting of a long scarf wrapped around the head",
        "frequency": "c",
        "id": 1148,
        "name": "turban",
        "synonyms": ["turban"],
        "synset": "turban.n.01"
    },
    {
        "def": "large gallinaceous bird with fan-shaped tail; widely domesticated for food",
        "frequency": "r",
        "id": 1149,
        "name": "turkey_(bird)",
        "synonyms": ["turkey_(bird)"],
        "synset": "turkey.n.01"
    },
    {
        "def": "flesh of large domesticated fowl usually roasted",
        "frequency": "c",
        "id": 1150,
        "name": "turkey_(food)",
        "synonyms": ["turkey_(food)"],
        "synset": "turkey.n.04"
    },
    {
        "def": "widely cultivated plant having a large fleshy edible white or yellow root",
        "frequency": "r",
        "id": 1151,
        "name": "turnip",
        "synonyms": ["turnip"],
        "synset": "turnip.n.01"
    },
    {
        "def": "any of various aquatic and land reptiles having a bony shell and flipper-like "
               "limbs for swimming",
        "frequency": "c",
        "id": 1152,
        "name": "turtle",
        "synonyms": ["turtle"],
        "synset": "turtle.n.02"
    },
    {
        "def": "a sweater or jersey with a high close-fitting collar",
        "frequency": "r",
        "id": 1153,
        "name": "turtleneck_(clothing)",
        "synonyms": ["turtleneck_(clothing)", "polo-neck"],
        "synset": "turtleneck.n.01"
    },
    {
        "def": "hand-operated character printer for printing written messages one character at a "
               "time",
        "frequency": "r",
        "id": 1154,
        "name": "typewriter",
        "synonyms": ["typewriter"],
        "synset": "typewriter.n.01"
    },
    {
        "def": "a lightweight handheld collapsible canopy",
        "frequency": "f",
        "id": 1155,
        "name": "umbrella",
        "synonyms": ["umbrella"],
        "synset": "umbrella.n.01"
    },
    {
        "def": "undergarment worn next to the skin and under the outer garments",
        "frequency": "c",
        "id": 1156,
        "name": "underwear",
        "synonyms": ["underwear", "underclothes", "underclothing", "underpants"],
        "synset": "underwear.n.01"
    },
    {
        "def": "a vehicle with a single wheel that is driven by pedals",
        "frequency": "r",
        "id": 1157,
        "name": "unicycle",
        "synonyms": ["unicycle"],
        "synset": "unicycle.n.01"
    },
    {
        "def": "a plumbing fixture (usually attached to the wall) used by men to urinate",
        "frequency": "c",
        "id": 1158,
        "name": "urinal",
        "synonyms": ["urinal"],
        "synset": "urinal.n.01"
    },
    {
        "def": "a large vase that usually has a pedestal or feet",
        "frequency": "r",
        "id": 1159,
        "name": "urn",
        "synonyms": ["urn"],
        "synset": "urn.n.01"
    },
    {
        "def": "an electrical home appliance that cleans by suction",
        "frequency": "c",
        "id": 1160,
        "name": "vacuum_cleaner",
        "synonyms": ["vacuum_cleaner"],
        "synset": "vacuum.n.04"
    },
    {
        "def": "control consisting of a mechanical device for controlling the flow of a fluid",
        "frequency": "c",
        "id": 1161,
        "name": "valve",
        "synonyms": ["valve"],
        "synset": "valve.n.03"
    },
    {
        "def": "an open jar of glass or porcelain used as an ornament or to hold flowers",
        "frequency": "f",
        "id": 1162,
        "name": "vase",
        "synonyms": ["vase"],
        "synset": "vase.n.01"
    },
    {
        "def": "a slot machine for selling goods",
        "frequency": "c",
        "id": 1163,
        "name": "vending_machine",
        "synonyms": ["vending_machine"],
        "synset": "vending_machine.n.01"
    },
    {
        "def": "a hole for the escape of gas or air",
        "frequency": "f",
        "id": 1164,
        "name": "vent",
        "synonyms": ["vent", "blowhole", "air_vent"],
        "synset": "vent.n.01"
    },
    {
        "def": "a video recording made on magnetic tape",
        "frequency": "c",
        "id": 1165,
        "name": "videotape",
        "synonyms": ["videotape"],
        "synset": "videotape.n.01"
    },
    {
        "def": "sour-tasting liquid produced usually by oxidation of the alcohol in wine or cider "
               "and used as a condiment or food preservative",
        "frequency": "r",
        "id": 1166,
        "name": "vinegar",
        "synonyms": ["vinegar"],
        "synset": "vinegar.n.01"
    },
    {
        "def": "bowed stringed instrument that is the highest member of the violin family",
        "frequency": "r",
        "id": 1167,
        "name": "violin",
        "synonyms": ["violin", "fiddle"],
        "synset": "violin.n.01"
    },
    {
        "def": "unaged colorless liquor originating in Russia",
        "frequency": "r",
        "id": 1168,
        "name": "vodka",
        "synonyms": ["vodka"],
        "synset": "vodka.n.01"
    },
    {
        "def": "an inflated ball used in playing volleyball",
        "frequency": "r",
        "id": 1169,
        "name": "volleyball",
        "synonyms": ["volleyball"],
        "synset": "volleyball.n.02"
    },
    {
        "def": "any of various large birds of prey having naked heads and weak claws and feeding "
               "chiefly on carrion",
        "frequency": "r",
        "id": 1170,
        "name": "vulture",
        "synonyms": ["vulture"],
        "synset": "vulture.n.01"
    },
    {
        "def": "pancake batter baked in a waffle iron",
        "frequency": "c",
        "id": 1171,
        "name": "waffle",
        "synonyms": ["waffle"],
        "synset": "waffle.n.01"
    },
    {
        "def": "a kitchen appliance for baking waffles",
        "frequency": "r",
        "id": 1172,
        "name": "waffle_iron",
        "synonyms": ["waffle_iron"],
        "synset": "waffle_iron.n.01"
    },
    {
        "def": "any of various kinds of wheeled vehicles drawn by an animal or a tractor",
        "frequency": "c",
        "id": 1173,
        "name": "wagon",
        "synonyms": ["wagon"],
        "synset": "wagon.n.01"
    },
    {
        "def": "a wheel of a wagon",
        "frequency": "c",
        "id": 1174,
        "name": "wagon_wheel",
        "synonyms": ["wagon_wheel"],
        "synset": "wagon_wheel.n.01"
    },
    {
        "def": "a stick carried in the hand for support in walking",
        "frequency": "c",
        "id": 1175,
        "name": "walking_stick",
        "synonyms": ["walking_stick"],
        "synset": "walking_stick.n.01"
    },
    {
        "def": "a clock mounted on a wall",
        "frequency": "c",
        "id": 1176,
        "name": "wall_clock",
        "synonyms": ["wall_clock"],
        "synset": "wall_clock.n.01"
    },
    {
        "def": "receptacle providing a place in a wiring system where current can be taken to run "
               "electrical devices",
        "frequency": "f",
        "id": 1177,
        "name": "wall_socket",
        "synonyms": ["wall_socket", "wall_plug", "electric_outlet", "electrical_outlet",
                     "outlet", "electric_receptacle"],
        "synset": "wall_socket.n.01"
    },
    {
        "def": "a pocket-size case for holding papers and paper money",
        "frequency": "c",
        "id": 1178,
        "name": "wallet",
        "synonyms": ["wallet", "billfold"],
        "synset": "wallet.n.01"
    },
    {
        "def": "either of two large northern marine mammals having ivory tusks and tough hide over "
               "thick blubber",
        "frequency": "r",
        "id": 1179,
        "name": "walrus",
        "synonyms": ["walrus"],
        "synset": "walrus.n.01"
    },
    {
        "def": "a tall piece of furniture that provides storage space for clothes; has a door and "
               "rails or hooks for hanging clothes",
        "frequency": "r",
        "id": 1180,
        "name": "wardrobe",
        "synonyms": ["wardrobe"],
        "synset": "wardrobe.n.01"
    },
    {
        "def": "the thick green root of the wasabi plant that the Japanese use in cooking and that "
               "tastes like strong horseradish",
        "frequency": "r",
        "id": 1181,
        "name": "wasabi",
        "synonyms": ["wasabi"],
        "synset": "wasabi.n.02"
    },
    {
        "def": "a home appliance for washing clothes and linens automatically",
        "frequency": "c",
        "id": 1182,
        "name": "automatic_washer",
        "synonyms": ["automatic_washer", "washing_machine"],
        "synset": "washer.n.03"
    },
    {
        "def": "a small, portable timepiece",
        "frequency": "f",
        "id": 1183,
        "name": "watch",
        "synonyms": ["watch", "wristwatch"],
        "synset": "watch.n.01"
    },
    {
        "def": "a bottle for holding water",
        "frequency": "f",
        "id": 1184,
        "name": "water_bottle",
        "synonyms": ["water_bottle"],
        "synset": "water_bottle.n.01"
    },
    {
        "def": "a device for cooling and dispensing drinking water",
        "frequency": "c",
        "id": 1185,
        "name": "water_cooler",
        "synonyms": ["water_cooler"],
        "synset": "water_cooler.n.01"
    },
    {
        "def": "a faucet for drawing water from a pipe or cask",
        "frequency": "c",
        "id": 1186,
        "name": "water_faucet",
        "synonyms": ["water_faucet", "water_tap", "tap_(water_faucet)"],
        "synset": "water_faucet.n.01"
    },
    {
        "def": "a filter to remove impurities from the water supply",
        "frequency": "r",
        "id": 1187,
        "name": "water_filter",
        "synonyms": ["water_filter"],
        "synset": "water_filter.n.01"
    },
    {
        "def": "a heater and storage tank to supply heated water",
        "frequency": "r",
        "id": 1188,
        "name": "water_heater",
        "synonyms": ["water_heater", "hot-water_heater"],
        "synset": "water_heater.n.01"
    },
    {
        "def": "a jug that holds water",
        "frequency": "r",
        "id": 1189,
        "name": "water_jug",
        "synonyms": ["water_jug"],
        "synset": "water_jug.n.01"
    },
    {
        "def": "plaything consisting of a toy pistol that squirts water",
        "frequency": "r",
        "id": 1190,
        "name": "water_gun",
        "synonyms": ["water_gun", "squirt_gun"],
        "synset": "water_pistol.n.01"
    },
    {
        "def": "a motorboat resembling a motor scooter (NOT A SURFBOARD OR WATER SKI)",
        "frequency": "c",
        "id": 1191,
        "name": "water_scooter",
        "synonyms": ["water_scooter", "sea_scooter", "jet_ski"],
        "synset": "water_scooter.n.01"
    },
    {
        "def": "broad ski for skimming over water towed by a speedboat (DO NOT MARK WATER)",
        "frequency": "c",
        "id": 1192,
        "name": "water_ski",
        "synonyms": ["water_ski"],
        "synset": "water_ski.n.01"
    },
    {
        "def": "a large reservoir for water",
        "frequency": "c",
        "id": 1193,
        "name": "water_tower",
        "synonyms": ["water_tower"],
        "synset": "water_tower.n.01"
    },
    {
        "def": "a container with a handle and a spout with a perforated nozzle; used to sprinkle "
               "water over plants",
        "frequency": "c",
        "id": 1194,
        "name": "watering_can",
        "synonyms": ["watering_can"],
        "synset": "watering_can.n.01"
    },
    {
        "def": "large oblong or roundish melon with a hard green rind and sweet watery red or "
               "occasionally yellowish pulp",
        "frequency": "c",
        "id": 1195,
        "name": "watermelon",
        "synonyms": ["watermelon"],
        "synset": "watermelon.n.02"
    },
    {
        "def": "mechanical device attached to an elevated structure; rotates freely to show the "
               "direction of the wind",
        "frequency": "f",
        "id": 1196,
        "name": "weathervane",
        "synonyms": ["weathervane", "vane_(weathervane)", "wind_vane"],
        "synset": "weathervane.n.01"
    },
    {
        "def": "a digital camera designed to take digital photographs and transmit them over the "
               "internet",
        "frequency": "c",
        "id": 1197,
        "name": "webcam",
        "synonyms": ["webcam"],
        "synset": "webcam.n.01"
    },
    {
        "def": "a rich cake with two or more tiers and covered with frosting and decorations; "
               "served at a wedding reception",
        "frequency": "c",
        "id": 1198,
        "name": "wedding_cake",
        "synonyms": ["wedding_cake", "bridecake"],
        "synset": "wedding_cake.n.01"
    },
    {
        "def": "a ring given to the bride and/or groom at the wedding",
        "frequency": "c",
        "id": 1199,
        "name": "wedding_ring",
        "synonyms": ["wedding_ring", "wedding_band"],
        "synset": "wedding_ring.n.01"
    },
    {
        "def": "a close-fitting garment made of a permeable material; worn in cold water to retain "
               "body heat",
        "frequency": "f",
        "id": 1200,
        "name": "wet_suit",
        "synonyms": ["wet_suit"],
        "synset": "wet_suit.n.01"
    },
    {
        "def": "a circular frame with spokes (or a solid disc) that can rotate on a shaft or axle",
        "frequency": "f",
        "id": 1201,
        "name": "wheel",
        "synonyms": ["wheel"],
        "synset": "wheel.n.01"
    },
    {
        "def": "a movable chair mounted on large wheels",
        "frequency": "c",
        "id": 1202,
        "name": "wheelchair",
        "synonyms": ["wheelchair"],
        "synset": "wheelchair.n.01"
    },
    {
        "def": "cream that has been beaten until light and fluffy",
        "frequency": "c",
        "id": 1203,
        "name": "whipped_cream",
        "synonyms": ["whipped_cream"],
        "synset": "whipped_cream.n.01"
    },
    {
        "def": "a liquor made from fermented mash of grain",
        "frequency": "r",
        "id": 1204,
        "name": "whiskey",
        "synonyms": ["whiskey"],
        "synset": "whiskey.n.01"
    },
    {
        "def": "a small wind instrument that produces a whistling sound by blowing into it",
        "frequency": "r",
        "id": 1205,
        "name": "whistle",
        "synonyms": ["whistle"],
        "synset": "whistle.n.03"
    },
    {
        "def": "a loosely woven cord in a candle or oil lamp that is lit on fire",
        "frequency": "r",
        "id": 1206,
        "name": "wick",
        "synonyms": ["wick"],
        "synset": "wick.n.02"
    },
    {
        "def": "hairpiece covering the head and made of real or synthetic hair",
        "frequency": "c",
        "id": 1207,
        "name": "wig",
        "synonyms": ["wig"],
        "synset": "wig.n.01"
    },
    {
        "def": "a decorative arrangement of pieces of metal or glass or pottery that hang together "
               "loosely so the wind can cause them to tinkle",
        "frequency": "c",
        "id": 1208,
        "name": "wind_chime",
        "synonyms": ["wind_chime"],
        "synset": "wind_chime.n.01"
    },
    {
        "def": "a mill that is powered by the wind",
        "frequency": "c",
        "id": 1209,
        "name": "windmill",
        "synonyms": ["windmill"],
        "synset": "windmill.n.01"
    },
    {
        "def": "a container for growing plants on a windowsill",
        "frequency": "c",
        "id": 1210,
        "name": "window_box_(for_plants)",
        "synonyms": ["window_box_(for_plants)"],
        "synset": "window_box.n.01"
    },
    {
        "def": "a mechanical device that cleans the windshield",
        "frequency": "f",
        "id": 1211,
        "name": "windshield_wiper",
        "synonyms": ["windshield_wiper", "windscreen_wiper", "wiper_(for_windshield/screen)"],
        "synset": "windshield_wiper.n.01"
    },
    {
        "def": "a truncated cloth cone mounted on a mast/pole; shows wind direction",
        "frequency": "c",
        "id": 1212,
        "name": "windsock",
        "synonyms": ["windsock", "air_sock", "air-sleeve", "wind_sleeve", "wind_cone"],
        "synset": "windsock.n.01"
    },
    {
        "def": "a bottle for holding wine",
        "frequency": "f",
        "id": 1213,
        "name": "wine_bottle",
        "synonyms": ["wine_bottle"],
        "synset": "wine_bottle.n.01"
    },
    {
        "def": "a bucket of ice used to chill a bottle of wine",
        "frequency": "r",
        "id": 1214,
        "name": "wine_bucket",
        "synonyms": ["wine_bucket", "wine_cooler"],
        "synset": "wine_bucket.n.01"
    },
    {
        "def": "a glass that has a stem and in which wine is served",
        "frequency": "f",
        "id": 1215,
        "name": "wineglass",
        "synonyms": ["wineglass"],
        "synset": "wineglass.n.01"
    },
    {
        "def": "easy chair having wings on each side of a high back",
        "frequency": "r",
        "id": 1216,
        "name": "wing_chair",
        "synonyms": ["wing_chair"],
        "synset": "wing_chair.n.01"
    },
    {
        "def": "blinds that prevent a horse from seeing something on either side",
        "frequency": "c",
        "id": 1217,
        "name": "blinder_(for_horses)",
        "synonyms": ["blinder_(for_horses)"],
        "synset": "winker.n.02"
    },
    {
        "def": "pan with a convex bottom; used for frying in Chinese cooking",
        "frequency": "c",
        "id": 1218,
        "name": "wok",
        "synonyms": ["wok"],
        "synset": "wok.n.01"
    },
    {
        "def": "a wild carnivorous mammal of the dog family, living and hunting in packs",
        "frequency": "r",
        "id": 1219,
        "name": "wolf",
        "synonyms": ["wolf"],
        "synset": "wolf.n.01"
    },
    {
        "def": "a spoon made of wood",
        "frequency": "c",
        "id": 1220,
        "name": "wooden_spoon",
        "synonyms": ["wooden_spoon"],
        "synset": "wooden_spoon.n.02"
    },
    {
        "def": "an arrangement of flowers, leaves, or stems fastened in a ring",
        "frequency": "c",
        "id": 1221,
        "name": "wreath",
        "synonyms": ["wreath"],
        "synset": "wreath.n.01"
    },
    {
        "def": "a hand tool that is used to hold or twist a nut or bolt",
        "frequency": "c",
        "id": 1222,
        "name": "wrench",
        "synonyms": ["wrench", "spanner"],
        "synset": "wrench.n.03"
    },
    {
        "def": "band consisting of a part of a sleeve that covers the wrist",
        "frequency": "c",
        "id": 1223,
        "name": "wristband",
        "synonyms": ["wristband"],
        "synset": "wristband.n.01"
    },
    {
        "def": "a band or bracelet worn around the wrist",
        "frequency": "f",
        "id": 1224,
        "name": "wristlet",
        "synonyms": ["wristlet", "wrist_band"],
        "synset": "wristlet.n.01"
    },
    {
        "def": "an expensive vessel propelled by sail or power and used for cruising or racing",
        "frequency": "r",
        "id": 1225,
        "name": "yacht",
        "synonyms": ["yacht"],
        "synset": "yacht.n.01"
    },
    {
        "def": "large long-haired wild ox of Tibet often domesticated",
        "frequency": "r",
        "id": 1226,
        "name": "yak",
        "synonyms": ["yak"],
        "synset": "yak.n.02"
    },
    {
        "def": "a custard-like food made from curdled milk",
        "frequency": "c",
        "id": 1227,
        "name": "yogurt",
        "synonyms": ["yogurt", "yoghurt", "yoghourt"],
        "synset": "yogurt.n.01"
    },
    {
        "def": "gear joining two animals at the neck; NOT egg yolk",
        "frequency": "r",
        "id": 1228,
        "name": "yoke_(animal_equipment)",
        "synonyms": ["yoke_(animal_equipment)"],
        "synset": "yoke.n.07"
    },
    {
        "def": "any of several fleet black-and-white striped African equines",
        "frequency": "f",
        "id": 1229,
        "name": "zebra",
        "synonyms": ["zebra"],
        "synset": "zebra.n.01"
    },
    {
        "def": "small cucumber-shaped vegetable marrow; typically dark green",
        "frequency": "c",
        "id": 1230,
        "name": "zucchini",
        "synonyms": ["zucchini", "courgette"],
        "synset": "zucchini.n.02"
    }
]
