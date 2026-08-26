"""Controlled vocabulary of canonical ingredient names.

Retrieval, diet inference and coverage scoring all key off these names rather than
raw recipe strings, so that "2 large ripe Roma tomatoes, chopped" and "tomato"
compare equal. Entries are singular and lowercase.
"""

VEGETABLES = frozenset(
    {
        "artichoke", "arugula", "asparagus", "avocado", "bamboo shoot", "bean sprout",
        "beet", "bell pepper", "bok choy", "broccoli", "brussels sprout", "cabbage",
        "carrot", "cauliflower", "celery", "chard", "chili pepper", "collard green", "corn",
        "cucumber",
        "eggplant", "garlic", "green bean", "green onion", "jalapeno", "kale", "leek",
        "lettuce", "mushroom", "okra", "olive", "onion", "parsnip", "pea", "pickle",
        "potato", "pumpkin", "radish", "rutabaga", "shallot", "snow pea", "spinach",
        "squash", "sweet potato", "tomato", "turnip", "water chestnut", "watercress",
        "zucchini",
    }
)

FRUITS = frozenset(
    {
        "apple", "apricot", "banana", "blackberry", "blueberry", "cantaloupe", "cherry",
        "coconut", "cranberry", "currant", "date", "fig", "grape", "grapefruit", "kiwi",
        "lemon", "lime", "mango", "nectarine", "orange", "papaya", "peach", "pear",
        "pineapple", "plum", "pomegranate", "prune", "raisin", "raspberry", "rhubarb",
        "strawberry", "tangerine", "watermelon",
    }
)

MEAT = frozenset(
    {
        "bacon", "beef", "bologna", "chicken", "chicken breast", "chicken thigh",
        "chicken wing", "chorizo", "duck", "ground beef", "ground pork", "ground turkey",
        "ham", "hot dog", "lamb", "liver", "meatball", "pastrami", "pepperoni", "pork",
        "pork chop", "prosciutto", "rabbit", "salami", "sausage", "steak", "turkey",
        "veal", "venison",
    }
)

SEAFOOD = frozenset(
    {
        "anchovy", "calamari", "clam", "cod", "crab", "fish", "haddock", "halibut",
        "lobster", "mussel", "octopus", "oyster", "salmon", "sardine", "scallop",
        "shrimp", "snapper", "sole", "squid", "tilapia", "trout", "tuna",
    }
)

DAIRY = frozenset(
    {
        "blue cheese", "brie", "butter", "buttermilk", "cheddar cheese", "cheese",
        "condensed milk", "cottage cheese", "cream", "cream cheese", "evaporated milk",
        "feta cheese", "ghee", "goat cheese", "gouda", "gruyere", "half and half",
        "heavy cream", "ice cream", "mascarpone", "milk", "monterey jack cheese",
        "mozzarella cheese", "paneer", "parmesan cheese", "provolone cheese",
        "ricotta cheese", "romano cheese", "sour cream", "swiss cheese",
        "whipped topping", "whipping cream", "yogurt",
    }
)

EGGS = frozenset({"egg", "egg white", "egg yolk", "mayonnaise"})

GRAINS = frozenset(
    {
        "all-purpose flour", "barley", "bran", "bread", "breadcrumb", "brown rice",
        "bulgur", "cake mix", "cereal", "cornmeal", "cornstarch", "couscous", "cracker",
        "croissant", "crouton", "english muffin", "granola", "grits", "lasagna noodle",
        "macaroni", "muffin", "noodle", "oat", "pasta", "phyllo dough", "pie crust",
        "pita bread", "pizza dough", "polenta", "puff pastry", "quinoa", "rice",
        "rice noodle", "roll", "rye flour", "self-rising flour", "semolina", "spaghetti",
        "tortilla", "wheat germ", "whole wheat flour", "wonton wrapper",
    }
)

LEGUMES_NUTS = frozenset(
    {
        "almond", "black bean", "cannellini bean", "cashew", "chestnut", "chia seed",
        "chickpea", "flaxseed", "hazelnut", "kidney bean", "lentil", "lima bean",
        "macadamia", "navy bean", "peanut", "peanut butter", "pecan", "pine nut",
        "pinto bean", "pistachio", "poppy seed", "pumpkin seed", "refried bean",
        "sesame seed", "soybean", "split pea", "sunflower seed", "walnut",
    }
)

SOY = frozenset({"edamame", "miso", "seitan", "soy sauce", "tempeh", "tofu"})

FATS = frozenset(
    {
        "canola oil", "coconut oil", "cooking spray", "corn oil", "lard", "margarine",
        "olive oil", "peanut oil", "sesame oil", "shortening", "vegetable oil",
    }
)

SWEETENERS = frozenset(
    {
        "agave", "brown sugar", "corn syrup", "honey", "maple syrup", "molasses",
        "powdered sugar", "stevia", "sugar",
    }
)

BAKING = frozenset(
    {
        "almond extract", "baking powder", "baking soda", "chocolate", "chocolate chip",
        "cocoa powder", "cream of tartar", "food coloring", "gelatin", "marshmallow",
        "pudding mix", "vanilla extract", "white chocolate", "yeast",
    }
)

CONDIMENTS = frozenset(
    {
        "apple cider vinegar", "balsamic vinegar", "barbecue sauce", "beef broth",
        "bouillon", "chicken broth", "chili sauce", "cream of chicken soup",
        "cream of mushroom soup", "enchilada sauce", "fish sauce", "gravy",
        "hoisin sauce", "horseradish", "hot sauce", "ketchup", "marinara sauce",
        "mustard", "oyster sauce", "pesto", "relish", "rice vinegar", "salad dressing",
        "salsa", "sriracha", "tahini", "teriyaki sauce", "tomato paste", "tomato sauce",
        "vegetable broth", "vinegar", "worcestershire sauce",
    }
)

SPICES = frozenset(
    {
        "allspice", "anise", "basil", "bay leaf", "black pepper", "caraway", "cardamom",
        "cayenne", "chili powder", "chive", "cilantro", "cinnamon", "clove", "coriander",
        "cumin", "curry powder", "dill", "fennel", "garlic powder", "ginger",
        "italian seasoning", "marjoram", "mint", "mustard seed", "nutmeg",
        "old bay seasoning", "onion powder", "oregano", "paprika", "parsley", "pepper",
        "poultry seasoning", "red pepper flake", "rosemary", "saffron", "sage", "salt",
        "tarragon", "thyme", "turmeric", "vanilla",
    }
)

LIQUIDS = frozenset(
    {
        "apple juice", "coffee", "cola", "cranberry juice", "ice", "lemon juice",
        "lime juice", "orange juice", "pineapple juice", "soda", "tea", "tomato juice",
        "water",
    }
)

ALCOHOL = frozenset(
    {
        "beer", "bourbon", "brandy", "champagne", "gin", "liqueur", "red wine", "rum",
        "sherry", "tequila", "vermouth", "vodka", "whiskey", "white wine", "wine",
    }
)

VOCABULARY = frozenset(
    VEGETABLES
    | FRUITS
    | MEAT
    | SEAFOOD
    | DAIRY
    | EGGS
    | GRAINS
    | LEGUMES_NUTS
    | SOY
    | FATS
    | SWEETENERS
    | BAKING
    | CONDIMENTS
    | SPICES
    | LIQUIDS
    | ALCOHOL
    | {"flour", "oil", "broth", "stock", "juice", "nut", "bean", "seed", "syrup"}
)

# Regional names, brand names common in scraped corpora, and frequent misspellings.
SYNONYMS = {
    "aubergine": "eggplant",
    "bicarbonate of soda": "baking soda",
    "caster sugar": "sugar",
    "castor sugar": "sugar",
    "confectioners sugar": "powdered sugar",
    "cool whip": "whipped topping",
    "corn flour": "cornstarch",
    "cornflour": "cornstarch",
    "courgette": "zucchini",
    "crisco": "shortening",
    "curd": "yogurt",
    "capsicum": "bell pepper",
    "beef stock": "beef broth",
    "chicken stock": "chicken broth",
    "vegetable stock": "vegetable broth",
    "chile": "chili pepper",
    "chili": "chili pepper",
    "serrano": "chili pepper",
    "habanero": "chili pepper",
    "garbanzo": "chickpea",
    "garbanzo bean": "chickpea",
    "green pepper": "bell pepper",
    "red pepper": "bell pepper",
    "sweet pepper": "bell pepper",
    "groundnut": "peanut",
    "hamburger": "ground beef",
    "hamburger meat": "ground beef",
    "icing sugar": "powdered sugar",
    "maize": "corn",
    "margerine": "margarine",
    "mince": "ground beef",
    "minced beef": "ground beef",
    "miracle whip": "mayonnaise",
    "oleo": "margarine",
    "plain flour": "all-purpose flour",
    "prawn": "shrimp",
    "rocket": "arugula",
    "scallion": "green onion",
    "soda pop": "soda",
    "soya": "soybean",
    "spring onion": "green onion",
    "swede": "rutabaga",
    "velveeta": "cheese",
    "white sugar": "sugar",
    "granulated sugar": "sugar",
    "sweet corn": "corn",
    "table salt": "salt",
    "kosher salt": "salt",
    "sea salt": "salt",
    "white flour": "all-purpose flour",
    "spring water": "water",
    "cold water": "water",
    "hot water": "water",
    "warm water": "water",
    "boiling water": "water",
    "olive oil spray": "cooking spray",
}

# Ingredients assumed present in almost any kitchen. A recipe is not treated as
# "unmakeable" because the user did not list these.
PANTRY_STAPLES = frozenset(
    {
        "all-purpose flour", "baking powder", "baking soda", "black pepper", "brown sugar",
        "cooking spray", "flour", "ice", "oil", "olive oil", "pepper", "salt", "sugar",
        "vegetable oil", "vinegar", "water",
    }
)
