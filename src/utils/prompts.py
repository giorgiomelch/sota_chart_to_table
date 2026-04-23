PROMPT_AreaLineBarHistogram = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.
In the presence of two Y-axes, exclusively extract the data series referring to the left axis and completely ignore the series referring to the right axis.

Return the output EXCLUSIVELY in valid JSON format, without any additional text or markdown formatting outside the JSON.
The JSON must strictly adhere to the following structure:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)",
    "categorical_axis": "Specify which axis represents the categories (independent variable). Answer EXCLUSIVELY with the string 'x' or 'y'. If the chart does not have a categorical axis (e.g., scatter plot with two numerical axes), return null.",
    "data_points": [
        {
        "series_name": "Name of the series (e.g., legend entry). Use 'Main' if there is only one series without a legend.",
        "x_value": "Category or numerical value on the X-axis.",
        "y_value": "Category or numerical value on the Y-axis."
        }
    ]
}
"""

PROMPT_Scatter = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided scatter plot image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.
In the presence of two Y-axes, exclusively extract the data series referring to the left axis and completely ignore the series referring to the right axis.

Return the output EXCLUSIVELY in valid JSON format, without any additional text or markdown formatting outside the JSON.
The JSON must strictly adhere to the following structure:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)",
    "data_points": [
        {
        "series_name": "Name of the series (e.g., legend entry). Use 'Main' if there is only one series without a legend.",
        "x_value": "Category or numerical value on the X-axis.",
        "y_value": "Numerical value on the Y-axis."
        }
    ]
}
"""

PROMPT_Radar = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart radar image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.

Return the output EXCLUSIVELY in valid JSON format, without any additional text or markdown formatting outside the JSON.
The JSON must strictly adhere to the following structure:

{
    "chart_title": "Main title of the chart (null if absent)",
    "data_points": [
        {
            "series_name": "Name of the series (e.g., legend entry). Use 'Main' if there is only one series.",
            "x_value": "Use the name of the vertex/variable.",
            "y_value": "Corresponding numerical value."
        }
    ]
}
"""

PROMPT_Pie = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart pie image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.

Return the output EXCLUSIVELY in valid JSON format, without any additional text or markdown formatting outside the JSON.
The JSON must strictly adhere to the following structure:

{
    "chart_title": "Main title of the chart (null if absent)",
    "data_points": [
        {
            "series_name": "Name of the series (e.g., legend entry). Use 'Main' if there is only one series.",
            "x_value": "Use the slice name.",
            "y_value": "Corresponding numerical value (percentage, count, or score)."
        }
    ]
}
"""

PROMPT_Box = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart box image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.
In the presence of two Y-axes, exclusively extract the data series referring to the left axis and completely ignore the series referring to the right axis.

Return the output EXCLUSIVELY in valid JSON format, without any additional text, preambles, or comments.
The JSON must strictly adhere to the following standardized structure, which adapts to different types of charts:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)",
    "categorical_axis": "Specify which axis represents the categories (independent variable). Answer EXCLUSIVELY with the string 'x' or 'y'. If the chart does not have a categorical axis (e.g., scatter plot with two numerical axes), return null.",
    "data_points": [
        {
            "series_name": "Name of the series or group. Use 'Main' if there is only one series.",
            "x_value": "Category or numerical value on the X-axis.",
            "y_value": {
                "min": "minimum value (if applicable, otherwise null)",
                "q1": "first quartile (if applicable, otherwise null)",
                "median": "median or central value (if applicable, otherwise null)",
                "q3": "third quartile (if applicable, otherwise null)",
                "max": "maximum value (if applicable, otherwise null)"
            }
        }
    ]
}
"""

PROMPT_Errorpoint = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart error-point image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.
In the presence of two Y-axes, exclusively extract the data series referring to the left axis and completely ignore the series referring to the right axis.

Return the output EXCLUSIVELY in valid JSON format, without any additional text, preambles, or comments.
The JSON must strictly adhere to the following standardized structure, which adapts to different types of charts:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)",
    "categorical_axis": "Specify which axis represents the categories (independent variable). Answer EXCLUSIVELY with the string 'x' or 'y'. If the chart does not have a categorical axis (e.g., scatter plot with two numerical axes), return null.",
    "data_points": [
        {
            "series_name": "Name of the series or group. Use 'Main' if there is only one series.",
            "x_value": "Category or numerical value on the X-axis.",
            "y_value": {
                "min": "minimum value (null if absent)",
                "median": "median or central value (null if absent)",
                "max": "maximum value (null if absent)"
            }
        }
    ]
}
"""

PROMPT_Bubble = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided chart bubble image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the axes.
In the presence of two Y-axes, exclusively extract the data series referring to the left axis and completely ignore the series referring to the right axis.

Return the output EXCLUSIVELY in valid JSON format, without any additional text, preambles, or comments.
The JSON must strictly adhere to the following standardized structure, which adapts to different types of charts:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)",
    "categorical_axis": "Specify which axis represents the categories (independent variable). Answer EXCLUSIVELY with the string 'x' or 'y'. If the chart does not have a categorical axis (e.g., scatter plot with two numerical axes), return null.",
    "data_points": [
        {
            "series_name": "Name of the series or legend entry. Use 'Main' if there is only one series.",
            "x_value": "Numerical value or category on the X-axis.",
            "y_value": "Numerical value or category on the Y-axis.",
            "z_value": "Numerical value represented by the bubble size.",
            "w_value": "Numerical value represented by the bubble color, null if the color indicates only the category"
        }
    ]
}
"""

PROMPT_Heatmap = """
You are an expert data analyst in extracting visual information. 
Your task is to analyze the provided heatmap image and reconstruct the underlying data table by extracting the exact values or, 
if not explicitly stated, making the most accurate estimate possible based on the colors.

Return the output EXCLUSIVELY in valid JSON format, without any additional text or markdown formatting outside the JSON.
The JSON must strictly adhere to the following structure:

{
    "chart_title": "Main title of the chart (null if absent)",
    "x_axis_label": "X-axis label (null if absent)",
    "y_axis_label": "Y-axis label (null if absent)", 
    "data_points": [
        {
        "x_value": "Category or numerical value on the X-axis.",
        "y_value": "Category or numerical value on the Y-axis.",
        "cell_value": "Numerical value represented by the number or the color intensity of the cell (null if absent)."
        }
    ]
}
"""

PROMPT2CHARTCLASS = {
    "area": PROMPT_AreaLineBarHistogram,
    "line": PROMPT_AreaLineBarHistogram,
    "bar": PROMPT_AreaLineBarHistogram,
    "heatmap": PROMPT_Heatmap,
    "histogram": PROMPT_AreaLineBarHistogram,      
    "scatter": PROMPT_Scatter,
    "radar": PROMPT_Radar,
    "pie": PROMPT_Pie,
    "box": PROMPT_Box,
    "errorpoint": PROMPT_Errorpoint,
    "bubble": PROMPT_Bubble,           
}