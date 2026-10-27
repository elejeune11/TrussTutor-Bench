"""
Central repository for recurring JSON schema definitions and
controlled vocabularies used across the benchmark.
"""

FORCE_PROPERTY_SCHEMA = {
    "type": "object",
    "properties": {
        "magnitude": {
            "type": "number",
            "description": "Absolute value of the internal force."
        },
        "state": {
            "type": "string",
            "enum": ["tension", "compression", "zero"],
            "description": "Physical state of the member."
        }
    },
    "required": ["magnitude", "state"],
    "additionalProperties": False
}

ALLOWED_TOPICS = [
    "method_of_joints",
    "method_of_sections",
    "identify_zfm",
    "find_unknown_load_or_dimension"
]

ALLOWED_SOURCE_TYPES = [
    "original",
    "textbook"
]

ALLOWED_SOURCE_NAMES = [
    "github test example",
    "Engineering Mechanics: Statics, Osgood et al.",
    "Engineering Statics Open and Interactive, Baker and Haynes",
    "elejeune11 handwritten",
    "sarajahedazad handwritten",
    "erfanhamdi handwritten",
    "eapostolakis99 handwritten",
    "Hibbeler 16th Edition 2026, Statics",
    "elejeune11 computer prepared, style 1",
    "elejeune11 computer prepared, style 2",
    "elejeune11 computer prepared, style 3",
    "erfanhamdi computer prepared, style 1",
    "erfanhamdi computer prepared, style 2",
    "erfanhamdi computer prepared, style 3",
    "erfanhamdi computer prepared, style 4",
    "erfanhamdi computer prepared, style 5",
    "Engineering Mechanics: Statics, 6th edition, Bedford & Fowler",
    "Vector Mechanics for Engineers: Statics, 12th Edition, by Ferdinand P. Beer, E. Russell Johnston Jr. and David F. Mazurek",
    "Introduction to Aerospace Structures and Materials. LibreTexts. Alderliesten, R."
]
