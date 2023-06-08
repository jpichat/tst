class FieldTypes:
    """Enum class for the supported field data types"""

    category = "category"
    numeric = "number"
    text = "text"
    nothing = "nothing"
    date = "date"

    @staticmethod
    def is_mixed(type):
        return type in (None, FieldTypes.numeric)

    @staticmethod
    def is_string(type):
        return type in (FieldTypes.category, FieldTypes.text, FieldTypes.date)

    @classmethod
    def is_valid(cls, field_type: str):
        return field_type in (
            cls.category,
            cls.numeric,
            cls.text,
            cls.nothing,
            cls.date,
        )
