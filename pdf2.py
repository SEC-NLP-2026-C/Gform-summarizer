import pandas as pd
import torch
from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
from fpdf import FPDF  # Import the FPDF library
import google.generativeai as genai
import tempfile  # To create temporary files for the charts
import os  # To remove temporary files
import ast
# Load the sentiment analysis model
sentiment = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                     torch_dtype=torch.bfloat16)





# Function to analyze sentiment of a single response
def analysis(response):
    output = sentiment(response)
    return output[0]['label']


# Function to read Excel file and return the dataframe along with column names
def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df, list(df.columns)  # Return both the dataframe and column names


# Function to create a pie chart of sentiments
def sentiment_pie_chart(df):
    sentiment_counts = df['Sentiments'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', color=['#8DA7E2','#a3d2ca'])
    ax.set_title('Review Sentiment Distribution')

    # Return the figure object
    return fig

def sentiment_donot_chart(df):
    # Data for the chart
    percentage = float(df[1])
    sizes = [percentage, (100-percentage)]  # Example data: Positive and Negative sentiments

    colors =['#8DA7E2','#a3d2ca']  # Mild colors for the segments

    # Create a pie chart
    fig, ax = plt.subplots()
    wedges = ax.pie(sizes,  colors=colors,  startangle=90)

    # Draw a circle at the center of the pie chart to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Display the percentage in the center
    total = sum(sizes)
    percentage = sizes[0] / total * 100
    ax.text(0, 0, f'{percentage:.1f}%', ha='center', va='center', fontsize=20)

    ax.set_title(str(df[0]))
    return fig



# Function to create a bar chart of sentiments
def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiments'].value_counts()

    # Create a bar chart
    fig, ax = plt.subplots()
    bars = sentiment_counts.plot(kind='bar', ax=ax, color=['#8DA7E2','#a3d2ca'])
    ax.set_title('Review Sentiment Counts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')

    # Annotate each bar with its count
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height()}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom')

    # Return the figure object
    return fig


# Function to perform sentiment analysis on the selected column
def process_sentiment(file_path, selected_column):
    df, _ = read_excel(file_path)
    if selected_column not in df.columns:
        raise ValueError(f"No Column named '{selected_column}'")

    df['Sentiments'] = df[selected_column].apply(analysis)
    pie_chart = sentiment_pie_chart(df)
    bar_chart = sentiment_bar_chart(df)
    return df, pie_chart, bar_chart


# Function to generate and download PDF report
def generate_pdf(file_path, selected_column):
    # Read the Excel file and apply sentiment analysis to the selected column
    df, _ = read_excel(file_path)
    df['Sentiments'] = df[selected_column].apply(analysis)

    # Initialize the Google Generative AI model
    llm = genai.GenerativeModel(model_name='gemini-pro')
    genai.configure(api_key='AIzaSyCr393tpEJumNQYoT3oYXcfl5WPCnSRe3Q')
    responses_para = list(df[selected_column].to_numpy())
    responses_para = " , ".join(responses_para)
    print(responses_para)
    # Generate responses based on the sample prompts
    res = llm.generate_content(
        '''
        Responses'''
        +
        str(responses_para)+
        '''
        

        

        what are the key parameters we can measure from these feedback responses. minimum - 3, maximum - 5. give the answer in a python list nested with list with parameters and their positive percentages(scale is 100, so give values from 0 to 100 only)  from the text, i don't need any other extra words or letters in the answer. The final list example format is given [
            ["Parameter1", value],
            ["Parameter2", value],
            ["Parameter3", value],
        ]
        '''
    )

    # Parse the response and get the list of parameters
    response_list = ast.literal_eval(res.text)
    print(response_list)
    response_description = llm.generate_content(
        '''
        give a short description that can be understood from the below content in 100 words without any title and give only text for the responses given as feedbacks.
        
        Responses
        '''
        +str(responses_para)
    )


    # Create a PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(190,  10, txt="Feedback Responses Report", ln=True, align='C', border = True)

    pie_chart = sentiment_pie_chart(df)
    pie_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pie_chart.savefig(pie_temp_file.name)
    pie_temp_file.close()  # Close the file explicitly
    plt.close(pie_chart)

    # Calculate chart positions
    x_position = 10 # Two columns: left (x=10) and right (x=110)
    y_position = 540   # Two charts per row, 70 points between rows

    # Insert the image into the PDF
    pdf.image(pie_temp_file.name, x=x_position, y=y_position, w=80)

    # If we've placed 5 charts, start a new page


    # Remove temporary files after they've been used
    os.remove(pie_temp_file.name)

    pie_chart = sentiment_bar_chart(df)
    pie_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pie_chart.savefig(pie_temp_file.name)
    pie_temp_file.close()  # Close the file explicitly
    plt.close(pie_chart)

    # Calculate chart positions
    x_position = 110  # Two columns: left (x=10) and right (x=110)
    y_position = 540   # Two charts per row, 70 points between rows

    # Insert the image into the PDF
    pdf.image(pie_temp_file.name, x=x_position, y=y_position, w=80)

    # If we've placed 5 charts, start a new page

    # Remove temporary files after they've been used
    os.remove(pie_temp_file.name)


    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    response_description = str(response_description.text).strip()
    pdf.multi_cell(170, 10, txt=response_description, align='L')
    pdf.ln(10)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(190, 10, txt="Feedback Responses Charts", ln=True, align='C', border=True)
    # Track chart position on page
    chart_count = 0

    # Save the charts to temporary files and insert them in the PDF
    for i in range(len(response_list)):
        pie_chart = sentiment_donot_chart(response_list[i])
        pie_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pie_chart.savefig(pie_temp_file.name)
        pie_temp_file.close()  # Close the file explicitly
        plt.close(pie_chart)

        # Calculate chart positions
        x_position = 10 if chart_count % 2 == 0 else 110  # Two columns: left (x=10) and right (x=110)
        y_position = 40 + (chart_count // 2) * 70  # Two charts per row, 70 points between rows

        # Insert the image into the PDF
        pdf.image(pie_temp_file.name, x=x_position, y=y_position, w=80)

        # If we've placed 5 charts, start a new page
        if chart_count == 5:
            pdf.add_page()
            chart_count = 0
        else:
            chart_count += 1

        # Remove temporary files after they've been used
        os.remove(pie_temp_file.name)

    # Create a temporary PDF file to store the output
    pdf_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(pdf_temp_file.name)

    return pdf_temp_file.name


# Main Gradio interface
def get_columns(file_path):
    _, columns = read_excel(file_path)
    return gr.Dropdown(choices=columns)


# Define the interface
with gr.Blocks() as demo:
    file_input = gr.File(file_types=["xlsx"], file_count='single', label="Upload your .xlsx file")

    # Dropdown for selecting the column (starts with a placeholder)
    column_dropdown = gr.Dropdown(label="Select the Responses Column after uploading file...", choices=[],
                                  interactive=True)

    # Output for displaying the resulting dataframe
    output_df = gr.Dataframe(label="Sentiments Predicted...")

    with gr.Row():
        output_pie_chart = gr.Plot(label="Pie Chart")
        output_bar_chart = gr.Plot(label="Bar Chart")

    # Update the dropdown immediately after file upload
    file_input.upload(fn=get_columns, inputs=file_input, outputs=column_dropdown)

    # Button to process the sentiment analysis
    analyze_button = gr.Button("Analyze Sentiments")

    # Connect the analyze button to the sentiment analysis function
    analyze_button.click(fn=process_sentiment, inputs=[file_input, column_dropdown],
                         outputs=[output_df, output_pie_chart, output_bar_chart])

    # Button to generate the PDF report
    generate_pdf_button = gr.Button("Generate PDF Report")

    # Connect the PDF generation function
    generate_pdf_button.click(fn=generate_pdf, inputs=[file_input, column_dropdown], outputs=gr.File())

# Launch the Gradio app
demo.launch()
