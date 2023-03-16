import math

from jsonschema import _types, _utils, _validators
from jsonschema.exceptions import ValidationError
from jsonschema.validators import create


def multipleOf(validator, dB, instance, schema):
    if not validator.is_type(instance, "number"):
        return

    if isinstance(dB, float):
        round_factor = int(abs(math.log10(dB)))
        quotient = instance / dB
        failed = round(quotient, round_factor) != round(quotient, round_factor)
    else:
        failed = instance % dB

    if failed:
        yield ValidationError("%r is not a multiple of %r" % (instance, dB))


def exclusiveMaximum(validator, maximum, instance, schema):
    if not validator.is_type(instance, "number") or isinstance(maximum, bool):
        return

    if instance >= maximum:
        yield ValidationError(
            "%r is greater than or equal to the maximum of %r"
            % (
                instance,
                maximum,
            ),
        )


def exclusiveMinimum(validator, minimum, instance, schema):
    if not validator.is_type(instance, "number") or isinstance(minimum, bool):
        return

    if instance <= minimum:
        yield ValidationError(
            "%r is less than or equal to the minimum of %r"
            % (
                instance,
                minimum,
            ),
        )


ParquetValidator = create(
    meta_schema=_utils.load_schema("draft7"),
    validators={
        u"$ref": _validators.ref,
        u"additionalItems": _validators.additionalItems,
        u"additionalProperties": _validators.additionalProperties,
        u"allOf": _validators.allOf,
        u"anyOf": _validators.anyOf,
        u"const": _validators.const,
        u"contains": _validators.contains,
        u"dependencies": _validators.dependencies,
        u"enum": _validators.enum,
        u"exclusiveMaximum": exclusiveMaximum,
        u"exclusiveMinimum": exclusiveMinimum,
        u"format": _validators.format,
        u"if": _validators.if_,
        u"items": _validators.items,
        u"maxItems": _validators.maxItems,
        u"maxLength": _validators.maxLength,
        u"maxProperties": _validators.maxProperties,
        u"maximum": _validators.maximum,
        u"minItems": _validators.minItems,
        u"minLength": _validators.minLength,
        u"minProperties": _validators.minProperties,
        u"minimum": _validators.minimum,
        u"multipleOf": multipleOf,
        u"oneOf": _validators.oneOf,
        u"not": _validators.not_,
        u"pattern": _validators.pattern,
        u"patternProperties": _validators.patternProperties,
        u"properties": _validators.properties,
        u"propertyNames": _validators.propertyNames,
        u"required": _validators.required,
        u"type": _validators.type,
        u"uniqueItems": _validators.uniqueItems,
    },
    type_checker=_types.draft7_type_checker,
    version="parquet",
)
