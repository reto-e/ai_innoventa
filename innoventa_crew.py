from langchain.llms import Ollama
from crewai import Agent

llm = Ollama(model="llama3")

# ----------------------------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------------------------

interviewer = Agent(
  role='Senior UX Researcher',
  goal='Prepare interview questions for a user interview about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "You are an experienced UX researcher."
    "You have conducted lots of user research interviews."
    "Hence you know what questions to ask to reveal need, wishes and requirements from potential customers."
  ),
  allow_delegation=False,
  llm = llm
)

potential_customer = Agent(
  role='Potential customer',
  goal='Tell the interviewer your experience, needs and wishes about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "You are an openminded, honest customer."
    "You are used to UX interviews. Thus with your answers to the question you reveal your thoughts and desires."
  ),
  allow_delegation=False,
  llm=llm
)

ux_converger = Agent(
  role='Senior UX Researcher',
  goal='out of the answers of the customer create a challenge statement.',
  verbose=True,
  memory=True,
  backstory=(
    "You are an experienced UX researcher."
    "You have conducted lots of user research interviews."
    "You are able to find the main problem out of the answers of a customer."
  ),
  allow_delegation=False,
  llm = llm
)

sw_engineer = Agent(
  role='Senior Software Engineer',
  goal='Create a variety of solutions for customers problem',
  verbose=True,
  memory=True,
  backstory=(
    "You are an experienced Software Engineer."
    "You are able to think of different solution approaches for a given customer challenge."
  ),
  allow_delegation=False,
  llm = llm
)

sw_architect = Agent(
  role='Senior Software Architect',
  goal='Create a list of screens for the solution of the solution_picker',
  verbose=True,
  memory=True,
  backstory=(
    "You are an experienced Software Architect."
    "You are able to break down a software solution into distinct screens and describe them."
  ),
  allow_delegation=False,
  llm = llm
)

sw_frontend_dev = Agent(
  role='Senior Software Frontend developer',
  goal='create html frontends for the screens that the sw_architect described.',
  verbose=True,
  memory=True,
  backstory=(
    "You are an experienced frontend dev."
    "You are able to create html files for the desired solution."
  ),
  allow_delegation=False,
  llm = llm
)

# ----------------------------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------------------------

from crewai import Task

# Research task
interview = Task(
  description=(
    "Prepare for a customer interview about {topic}."
    "Identify questions that will reveal the potential customer's needs, wishes and requirements."
    "Write down 5 questions you will ask the potential customer."
  ),
  expected_output='A list of 5 questions you will ask a potential customer.',
  agent=interviewer,
  output_file='ux_questions.md',  
  verbose=True
)

# Writing task with language model configuration
answer_questions = Task(
  description=(
    "Answer the questions asked by the interviewer"
    "Think out aloud, explain your reasoning for your answers."
  ),
  expected_output='Answers to the questions about {topic} including background information formatted as markdown.',
  agent=potential_customer,
  async_execution=False,
  output_file='customer_answers.md',  
  verbose=True
)

create_challenge = Task(
  description=(
    "Read the answers from the customer."
    "Formulate a challenge statement in the style of How might we help our customer to <solve main problem>"
    "where <solve main problem> is replaced with the problem you want to solve."
  ),
  expected_output='a challenge statement in the format of How might we help our customer to <solve main problem>',
  agent=ux_converger,
  async_execution=False,
  output_file='challenge.md',  
  verbose=True
)

collect_sollutions = Task(
  description=(
    "Read the challenge statement from the ux_converger."
    "List 10 different ideas for an app how you would solve the challenge."
    "Store the output as a markdown file."
  ),
  expected_output='a list of 10 ideas how to solve the challenge from the ux_converger',
  agent=sw_engineer,
  async_execution=False,
  output_file='solution_list.md',  
  verbose=True
)

generate_prototype_prompt = Task(
  description=(
    "Pick the best of the ideas of the solution_list."
    "Describe the goal and three key features of your app as a prompt for wiregen."
    "Use max 300 characters."
  ),
  expected_output='a prompt for wiregen with 500 characters.',
  agent=solution_picker,
  async_execution=False,
  output_file='solution_prompt.md',  
  verbose=True
)

solution_break_down = Task(
  description=(
    "Pick the best of the ideas of the solution_list."
    "Break this solution down into single screens that can be created using html."
    "For each screan create a name and a list of features."
  ),
  expected_output='a list of screens with their names and features.',
  agent=sw_architect,
  async_execution=False,
  output_file='solution_breakdown.md',  
  verbose=True
)

build_prototype = Task(
  description=(
    "Use the solution breakdown."
    "For each screen create an html file with the described features. "
    "Store those files as html files."
  ),
  expected_output='html files for each screen.',
  agent=sw_frontend_dev,
  async_execution=False,  
  verbose=True
)

# ----------------------------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------------------------

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[interviewer, potential_customer, ux_converger, sw_engineer, sw_architect, sw_frontend_dev],
  tasks=[interview, answer_questions, create_challenge, collect_sollutions, generate_prototype_prompt, solution_break_down, build_prototype],
  process=Process.sequential  # Optional: Sequential task execution is default
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'plan a barbecue party'})
print(result)
