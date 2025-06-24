"""
Load data for the culinary assistant
"""

from typing import List, Dict
from src.data import combine_external_data, get_external_data_stats, validate_data_files
from src.utils.logger import logger


def create_culinary_dataset() -> List[str]:
    """Create the culinary dataset with external data or fallback to base texts."""
    external_texts = combine_external_data()

    if external_texts:
        logger.info(f"Loaded {len(external_texts)} texts from external data files")
        return external_texts

    logger.warning("No external data found, using base texts")
    base_texts = [
        "To prepare a delicious ratatouille, cut the vegetables into regular slices",
        "Traditional veal blanquette requires gentle and long cooking",
        "The secret of a creamy risotto is the gradual addition of hot broth",
        "A successful shortcrust pastry requires working quickly with cold butter",
        "Perfect macarons require precision in measurements",
        "Béarnaise sauce is prepared delicately in a bain-marie",
        "Beef bourguignon simmers slowly in red wine",
        "A cheese soufflé requires stiffly beaten egg whites",
        "Crème anglaise is ready when it coats the spoon",
        "Roasted vegetables in the oven develop caramelized flavors",
        "Homemade bread requires good kneading and patience",
        "Light chocolate mousse gently incorporates the whites",
        "An authentic quiche lorraine uses cream and bacon",
        "Duck confit is preserved in its own fat",
        "Balanced vinaigrette respects oil vinegar proportions",
        "Fresh pasta cooks quickly in salted boiling water",
        "Chocolate fondant has a runny center after precise cooking",
        "Caramelizing onions requires patience and low heat",
        "Homemade mayonnaise is made by slowly adding oil",
        "A good pot-au-feu simmers with seasonal vegetables",
        "Blanching vegetables preserves their bright color",
        "Deglazing a pan recovers flavorful juices",
        "Marinating meat tenderizes and flavors delicately",
        "Clarified butter withstands high temperatures",
        "Reducing a sauce concentrates flavors intensely",
        "Poached eggs require simmering vinegared water",
        "Braising combines searing and simmering to tenderize",
        "A successful emulsion harmoniously binds ingredients",
        "Flaming evaporates alcohol while keeping aroma",
        "Sautéing over high heat sears and preserves textures",
    ]

    expanded_dataset = []
    for text in base_texts:
        expanded_dataset.append(text)

        words = text.split()
        if len(words) > 8:
            mid_point = len(words) // 2
            expanded_dataset.append(" ".join(words[:mid_point]))
            expanded_dataset.append(" ".join(words[mid_point:]))

    return expanded_dataset


def create_validation_dataset() -> List[str]:
    """Create a separate validation dataset from external data or base texts."""
    from src.data import load_cooking_tips_from_csv
    validation_texts = load_cooking_tips_from_csv()

    if validation_texts:
        return validation_texts

    validation_texts = [
        "Steam cooking preserves vegetable vitamins",
        "Good chicken broth requires bones and vegetables",
        "Choux pastry is prepared in two distinct steps",
        "Salted butter caramel is a Breton specialty",
        "Fresh herbs delicately flavor dishes",
        "Hollandaise sauce is made in a bain-marie",
        "Fish en papillote keeps all its flavor",
        "Pastry cream must be well cooled before use",
    ]

    return validation_texts


def get_dataset_stats(texts: List[str]) -> Dict:
    """Calculate dataset statistics."""
    total_words = sum(len(text.split()) for text in texts)
    avg_length = total_words / len(texts) if texts else 0
    max_length = max(len(text.split()) for text in texts) if texts else 0
    min_length = min(len(text.split()) for text in texts) if texts else 0

    return {
        "num_texts": len(texts),
        "total_words": total_words,
        "avg_length": avg_length,
        "max_length": max_length,
        "min_length": min_length,
    }


def get_data_source_info() -> Dict:
    """Get information about data sources."""
    external_stats = get_external_data_stats()
    validation = validate_data_files()
    
    return {
        "external_data_stats": external_stats,
        "file_validation": validation,
        "using_external_data": external_stats["total_files"] > 0
    }
