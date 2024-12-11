# env and libraries
import os
import litellm
import streamlit as st
import re
import pandas as pd
from io import StringIO
import json
import ast
import plotly.express as px
from streamlit_plotly_events import plotly_events
import re
import requests
from bs4 import BeautifulSoup
from readability import Document
import requests
from streamlit_card import card

# deployment issues
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

def extract_main_content_from_url(url):
    # Fetch the HTML content from the URL
    response = requests.get(url)
    html_content = response.content

    # Use Readability to extract the main content
    doc = Document(html_content)
    
    # Parse the main content using BeautifulSoup
    soup = BeautifulSoup(doc.summary(), 'lxml')

    # Extract the text from <p>, <h1>, <h2>, etc.
    main_content = ' '.join([element.get_text() for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])

    # Clean up extra whitespaces
    return ' '.join(main_content.split())


def google_search(query):
    """
    Perform a search using Google Custom Search API.

    :param api_key: Your Google API key.
    :param cse_id: Your Custom Search Engine ID.
    :param query: The search query string.
    :return: The JSON response from the API.
    """
    api_key = st.secrets["api_keys"]["google_api_key"]
    cse_id = st.secrets["api_keys"]["google_cse_id"]
    
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query
    }
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:

      results = response.json()

      result_string = ""
      if results:
            for item in results.get('items', []):
                result_string += f"Title: {item.get('title')}\n"
                result_string += f"Link: {item.get('link')}\n"
                result_string += f"Snippet: {item.get('snippet')}\n"
                result_string += f"Content: {extract_main_content_from_url(url)}\n"
                result_string += "|" + "\n"
      else:
            result_string = "No results found."
      return result_string

    else:
        # Handle errors
        print(f"Error: {response.status_code}")
        return None

pipeline = {
    0: "task_input",
    1: "decomposition",
    2: "approaches_and_criteria"
}

if "first" not in st.session_state:
    st.session_state["first"] = True

if "TASK" not in st.session_state:
    st.session_state["TASK"] = None

if "SUB_TASK" not in st.session_state:
    st.session_state["SUB_TASK"] = None

if "APPROACHES" not in st.session_state:
    st.session_state["APPROACHES"] = None

if "CRITERIA" not in st.session_state:
    st.session_state["CRITERIA"] = None

if "INITAL_ANSWERS" not in st.session_state:
    st.session_state["INITAL_ANSWERS"] = None

if "LINKS" not in st.session_state:
    st.session_state["LINKS"] = None

if 'RECOMMENDATIONS' not in st.session_state:
    st.session_state["RECOMMENDATIONS"] = None

if 'USER_PREFERENCE' not in st.session_state:
    st.session_state["USER_PREFERENCE"] = None

if 'PREFERRED_APPROACH' not in st.session_state:
    st.session_state["PREFERRED_APPROACH"] = None

if 'USER_INFERENCE' not in st.session_state:
    st.session_state["USER_INFERENCE"] = []

if 'step' not in st.session_state:
    st.session_state.step = 0


if 'continue' not in st.session_state:
    st.session_state['continue'] = True

# MODEL
# Set up the LiteLLM client
api_key = st.secrets["api_keys"]["litellm_api_key"] # ND key
base_url = "https://cmu.litellm.ai"
# model = "openai/gpt-4o"
gpt_model = "gpt-4o-mini"
model = f"openai/{gpt_model}" # Find ALL OpenAI model names here: https://platform.openai.com/docs/models

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = {
                                    "decomposition": "Someone is asking you {task}, ask a question to find out which part of the task " \
                                                    "they need help with and also include an option for they need help with everything and have " \
                                                    "never done this before.",
                                    "approach": "The user needs help with the following part: {sub_task} " \
                                                    "for the task {task}. Give me a json with all of the approaches for doing {task} " \
                                                    "and the expected outcome taking this approach in the format {{approaches: [{{approach: 'approach_1', description: 'description_1', outcome: 'outcome_1'}}, ...]}}. " \
                                                    "Consider every possible approach possible that a person could take that would account for the entire space in this task.",
                                    "criteria": "Given these approaches {approaches} for the task {task}, " \
                                                        "give me a json with all of the criteria that can be used to distinguish between approaches in the format " \
                                                        "{{criteria: [{{criteria: 'criteria_1', description: 'description_1'}}, ...]}}." \
                                                        " Order the criteria from most important to least important to the task.",
                                    "inital_questions": "Given these criteria: {criteria} and the approaches: {approaches} for the task: {task}, " \
                                                        "ask three questions about the top that the user can answer to give a recommended approach for the task.",
                                    "initial_recommendation": "Given these answers {answers} "
                                                        "and the approaches: {approaches} "
                                                        "for the task: {task}, " \
                                                        "give an initial recommendation to the user along with three other approaches they could take to achieve this goal" \
                                                        " by varying the following criteria: {criteria}. " \
                                                        "Here are some links that you can use to supplement the recommendations. Make sure to only use the referenced links {links}. " \
                                                        "Give me the recommendations in the following format {{recommendations: [{{recommendation: 'recommendation_1', description: 'description_1'', url: 'url_1'}}, ...]}}. " \
                                                        "Provide a lot of detail about the recommendations in the description and the tradeoffs regarding the criteria given above.",
                                    "user_preferences": "A user prefers this approach: {preferred_approach} for this task: {task} out of the following approaches {approaches}. What can you infer about the user preferences for these {criteria}? " \
                                                        "Provide the user's preferences for each of the criteria in a short description such as budget: high.",
                                    "generate_recommendations": "This is the user's current pick: {preferred_approach} for this task: {task}. The following are inferences about the user's preferences for these {criteria}: {inferences}. " \
                                                                # "Here are some links that you can use to supplement the recommendations. Make sure to only use the referenced links {links}." \
                                                                # "Do not use any of the approaches not preferred by the user {recommendations}. "
                                                                "Give me a recommendation and along with three other approaches that haven't been given they could take to achieve this goal in the following " \
                                                                "format {{recommendations: [{{recommendation: 'recommendation_1', description: 'description_1'', url: 'url_1'}}, ...]}}.",
                                    "curate_learning_path": "Develop a learning path for a person where this is their preferred approach {preferred_approach} for the task {task}. " \
                                                            "Keep in mind these inferences {inferences} about the user when creating the path and outline the outcome they will achieve after each step and the time each step will take. " \
                                                            "Give me specific places / options that I can go to in order to achieve this goal within {preferred_approach}. " \
                                                            "Here are some links that you can use. Make sure to only use the referenced links {links}."
                                }

systemt_context = "You are designed to assist users in identifying their learning needs by" \
                    "presenting various options tailored to their goals." \
                    "Your task is to guide users through a structured exploration of learning techniques, " \
                    "highlighting the different factors once should consider when trying to learn a specific skill."

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": f"{systemt_context}"} # change the initial system prompt
    ]

# Fixed send_to_llm function to return the response
def send_to_llm(messages):
    response = litellm.completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.2,  # Set temperature to 0 for deterministic responses
        max_tokens=1000
    )
    return response["choices"][0]["message"]["content"]

def parse_json_response(response):
    # Regular expression to capture the JSON block
    index_json = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    json_data = index_json.group(1).strip()  # Extracting the content inside the backticks

    extracted_dict = json.loads(json_data)

    return extracted_dict

# Step 1: Enter Task
def enter_task():
    initial_prompt = "What are you trying to learn?"
    st.session_state.messages.append({"role": "assistant", "content": initial_prompt})
    st.chat_message("assistant").markdown(initial_prompt)


# Step 2: Decomposition - Sub-tasks
def decomposition():
    decomposition_prompt = st.session_state["pipeline"]["decomposition"].format(task=st.session_state["TASK"])
    st.session_state['messages'].append({"role": "assistant", "content": decomposition_prompt})  # Assistant prompt
    subtasks = send_to_llm(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": subtasks})  # Assistant response
    st.chat_message("assistant").markdown(subtasks)

# Step 3: Approaches and Criteria
def approaches_and_criteria():
    approaches_prompt = st.session_state["pipeline"]["approach"].format(sub_task=st.session_state["SUB_TASK"], task=st.session_state["TASK"])
    st.session_state['messages'].append({"role": "assistant", "content": approaches_prompt})  # Assistant prompt
    approaches = send_to_llm(st.session_state.messages)
    approaches_dict = parse_json_response(approaches)
    st.session_state["APPROACHES"] = approaches_dict
    print(approaches_dict)
    st.session_state.messages.append({"role": "assistant", "content": approaches})  # Assistant response
    
    criteria_prompt = st.session_state["pipeline"]["criteria"].format(approaches=str(st.session_state["APPROACHES"]), task=st.session_state["TASK"])
    st.session_state['messages'].append({"role": "assistant", "content": criteria_prompt})  # Assistant prompt
    criteria = send_to_llm(st.session_state.messages)
    criteria_dict = parse_json_response(criteria)
    st.session_state["CRITERIA"] = [c for c in criteria_dict.keys()]
    st.session_state.messages.append({"role": "assistant", "content": criteria})  # Assistant response

def initial_questions():
    initial_questions_prompt = st.session_state["pipeline"]["inital_questions"].format(criteria=str(st.session_state["CRITERIA"][0:3]), 
                                                                                       approaches=str(st.session_state["APPROACHES"]),
                                                                                       task=st.session_state["TASK"])
    st.session_state['messages'].append({"role": "assistant", "content": initial_questions_prompt})  # Assistant prompt
    questions = send_to_llm(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": questions})  # Assistant response
    st.chat_message("assistant").markdown(questions)

def initial_recommendation():
    initial_recommendation_prompt = st.session_state["pipeline"]["initial_recommendation"].format(answers=st.session_state["INITAL_ANSWERS"], 
                                                                                       approaches=str(st.session_state["APPROACHES"]),
                                                                                       task=st.session_state["TASK"],
                                                                                       criteria=str(st.session_state["CRITERIA"]),
                                                                                       links=st.session_state["LINKS"]
                                                                                       )
    st.session_state['messages'].append({"role": "assistant", "content": initial_recommendation_prompt})  # Assistant prompt
    recommendation = send_to_llm(st.session_state.messages)
    st.session_state["RECOMMENDATIONS"] =  parse_json_response(recommendation)
    recommendation += "\n Enter 'recommendation' if you prefer the recommended approach or 1, 2, 3 if you prefer one of the alternatives."
    i = 0
    for doc_info in st.session_state["RECOMMENDATIONS"]['recommendations']:
        card_title = "Recommendation" if i == 0 else f"Alternative Approach {i}"
        # st.chat_message("assistant").markdown("Recommendation") if i == 0 else st.chat_message("assistant").markdown(f"Alternatve Approach {i}")
        has_clicked = card(
            title=card_title,
            text=f"{doc_info['recommendation']}: {doc_info['description']}",
            # url=doc_info['url'],
                    styles={
                "card": {
                    "width": "100%",
                    "height": "400px",
                    "border-radius": "2px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                }
            }
        )
        i += 1
        
    st.chat_message("assistant").markdown("Enter 'recommendation' if you prefer the recommended approach or 1, 2, 3 if you prefer one of the alternatives.")

def get_preferred_approach():
    
    recommedation_list = st.session_state["RECOMMENDATIONS"]['recommendations']

    if st.session_state["USER_PREFERENCE"] == "recommendation":
        return (recommedation_list[0], True)
    elif st.session_state["USER_PREFERENCE"] == "1":
        return (recommedation_list[1], False)
    elif st.session_state["USER_PREFERENCE"] == "2":
        return (recommedation_list[2], False)
    elif st.session_state["USER_PREFERENCE"] == "3":
        return (recommedation_list[3], False)
    else:
        return (recommedation_list[0], False)
    

def continue_with_approaches():
    st.session_state.messages.append({"role": "assistant", "content": "Do you want to see alternative approaches? Answer yes or no."})  # Assistant response
    st.chat_message("assistant").markdown("Do you want to see alternative approaches? Answer yes or no.")
    
def infer_user_preferences():
    inference_prompt = st.session_state["pipeline"]["user_preferences"].format(preferred_approach=st.session_state["PREFERRED_APPROACH"],
                                                                       task=st.session_state["TASK"],
                                                                       approaches=str(st.session_state["APPROACHES"]),
                                                                       criteria=str(st.session_state["CRITERIA"]))
    st.session_state['messages'].append({"role": "assistant", "content": inference_prompt})  # Assistant prompt
    inference = send_to_llm(st.session_state.messages)
    st.session_state["USER_INFERENCE"] = inference
    print(inference)

def generate_recommendations():
    recommendation_prompt = st.session_state["pipeline"]["generate_recommendations"].format(preferred_approach=st.session_state["PREFERRED_APPROACH"],
                                                                                   inferences=str(st.session_state["USER_INFERENCE"]),
                                                                                   task=st.session_state["TASK"],
                                                                                   recommendations=str(st.session_state["RECOMMENDATIONS"]),
                                                                                   criteria=str(st.session_state["CRITERIA"]),
                                                                                   links=st.session_state["LINKS"],
                                                                                   approaches=st.session_state["APPROACHES"]
                                                                                   )
    st.session_state['messages'].append({"role": "assistant", "content": recommendation_prompt})  # Assistant prompt
    recommendation = send_to_llm(st.session_state.messages)
    st.session_state["RECOMMENDATIONS"] =  parse_json_response(recommendation)
    i = 0
    for doc_info in st.session_state["RECOMMENDATIONS"]['recommendations']:
        card_title = "Recommendation" if i == 0 else f"Alternative Approach {i}"
        # st.chat_message("assistant").markdown("Recommendation") if i == 0 else st.chat_message("assistant").markdown(f"Alternatve Approach {i}")
        has_clicked = card(
            title=card_title,
            text=f"{doc_info['recommendation']}: {doc_info['description']}",
            url=doc_info['url'],
                    styles={
                "card": {
                    "width": "100%",
                    "height": "400px",
                    "border-radius": "10px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                }
            }
        )
        i += 1
    # recommendation += "\n Enter 'recommendation' if you prefer the recommended approach or 1, 2, 3 if you prefer one of the alternatives."
    st.session_state.messages.append({"role": "assistant", "content": recommendation})  # Assistant response
    st.chat_message("assistant").markdown("Enter 'recommendation' if you prefer the recommended approach or 1, 2, 3 if you prefer one of the alternatives.")

def curate_learning_path():
    learning_path_prompt = st.session_state["pipeline"]["curate_learning_path"].format(preferred_approach=st.session_state["PREFERRED_APPROACH"],
                                                                                   inferences=str(st.session_state["USER_INFERENCE"]),
                                                                                   task=st.session_state["TASK"],
                                                                                   links=st.session_state["LINKS"]
                                                                                   )

    st.session_state['messages'].append({"role": "assistant", "content": learning_path_prompt})  # Assistant prompt
    learning_path = send_to_llm(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": learning_path})  # Assistant response
    st.chat_message("assistant").markdown(learning_path)

def get_next_prompt():
    if st.session_state.step == 0:
        enter_task()  # Show task prompt
        st.session_state.step = 1
    elif st.session_state.step == 1:
        decomposition()  # Move to decomposition
        st.session_state.step = 2
    elif st.session_state.step == 2:
        approaches_and_criteria()  # Move on to approaches and criteria
        initial_questions()
        st.session_state.step = 3
    elif st.session_state.step == 3:
        initial_recommendation()
        st.session_state.step = 4
    elif st.session_state.step == 4:
        preferred_approach, is_recommended = get_preferred_approach()
        st.session_state["PREFERRED_APPROACH"] = preferred_approach
        print("getting preferences")
        if is_recommended:
            st.session_state.step = 5
            continue_with_approaches()
        else:
            st.session_state.step = 4
            infer_user_preferences()
            generate_recommendations()
    elif st.session_state.step == 5:
        # ask the user if they want to see alternative approaches
        continue_with_approaches()
    elif st.session_state.step == 6:
        # infer about the user's preferences
        print("Inferring user preferences")
        infer_user_preferences()
        st.session_state.step = 7
    elif st.session_state.step == 7:
        print("generating recommendations")
        generate_recommendations()
        st.session_state.step = 4
    elif st.session_state.step == 8:
        curate_learning_path()
        st.stop()


# Main code to handle user input
if user_response := st.chat_input("Enter response here"):
    st.chat_message("user").markdown(user_response)
    # Add user message to chat history before next step
    st.session_state.messages.append({"role": "user", "content": user_response})

    if st.session_state.step == 1:
        st.session_state["TASK"] = user_response  # Store the user's task input
        st.session_state["LINKS"] = google_search(st.session_state["TASK"])
    elif st.session_state.step == 2:
        st.session_state["SUB_TASK"] = user_response  # Store the user's sub-task input
    elif st.session_state.step == 3:
        st.session_state["INITAL_ANSWERS"] = user_response  # Store the user's initial questions input
    elif st.session_state.step == 4:
        st.session_state["USER_PREFERENCE"] = user_response  # Store the user's preference to the recommendations
    elif st.session_state.step == 5:
        if user_response.lower() == "yes":
            st.session_state.step = 7
        else:
            st.session_state.step = 8
            search_query = str(st.session_state["TASK"]) + " " + str(st.session_state["PREFERRED_APPROACH"])
            st.session_state["LINKS"] = google_search(search_query)
    
    # Call the next step in the process
    get_next_prompt()
