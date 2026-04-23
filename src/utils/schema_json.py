AREA_LINE_BAR_HISTOGRAM_SCHEMA = {
    "name": "area_line_bar_histogram_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "categorical_axis": {
                "anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]
            },
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "y_value": {"anyOf": [{"type": "string"}, {"type": "number"}]}
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "categorical_axis",
            "data_points"
        ]
    }
}
SCATTER_SCHEMA = {
    "name": "scatter_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                        "y_value": {"type": "number"}
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "data_points"]
    }
}
RADAR_SCHEMA = {
    "name": "radar_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"type": "string"},
                        "y_value": {"type": "number"}
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": ["chart_title", "data_points"]
    }
}
PIE_SCHEMA = {
    "name": "pie_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"type": "string"},
                        "y_value": {"type": "number"}
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": ["chart_title", "data_points"]
    }
}
BOX_SCHEMA = {
    "name": "box_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "categorical_axis": {
                "anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]
            },
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "y_value": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "q1": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "median": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "q3": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "max": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                            },
                            "required": ["min", "q1", "median", "q3", "max"]
                        }
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "categorical_axis",
            "data_points"
        ]
    }
}
ERRORPOINT_SCHEMA = {
    "name": "errorpoint_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "categorical_axis": {
                "anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]
            },
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "y_value": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "median": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "max": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                            },
                            "required": ["min", "median", "max"]
                        }
                    },
                    "required": ["series_name", "x_value", "y_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "categorical_axis",
            "data_points"]
    }
}
BUBBLE_SCHEMA = {
    "name": "bubble_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "categorical_axis": {
                "anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]
            },
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "series_name": {"type": "string"},
                        "x_value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                        "y_value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                        "z_value": {"type": "number"},
                        "w_value": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                    },
                    "required": ["series_name", "x_value", "y_value", "z_value", "w_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "categorical_axis",
            "data_points"]
    }
}
HEATMAP_SCHEMA = {
    "name": "heatmap_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "x_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "y_axis_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "y_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "cell_value": {"type": "number"}
                    },
                    "required": ["x_value", "y_value"]
                }
            }
        },
        "required": [
            "chart_title",
            "x_axis_label",
            "y_axis_label",
            "data_points"]
    }
}
SCHEMA2CHARTCLASS = {
    "area": AREA_LINE_BAR_HISTOGRAM_SCHEMA,
    "line": AREA_LINE_BAR_HISTOGRAM_SCHEMA,
    "bar": AREA_LINE_BAR_HISTOGRAM_SCHEMA,
    "histogram": AREA_LINE_BAR_HISTOGRAM_SCHEMA,
    "scatter": SCATTER_SCHEMA,
    "radar": RADAR_SCHEMA,
    "pie": PIE_SCHEMA,
    "box": BOX_SCHEMA,
    "errorpoint": ERRORPOINT_SCHEMA,
    "bubble": BUBBLE_SCHEMA,
    "heatmap": HEATMAP_SCHEMA
}
