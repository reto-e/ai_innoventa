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

# Creating a writer agent with custom tools and delegation capability
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
  agent=potential_customer,
  async_execution=False,
  output_file='challenge.md',  
  verbose=True
)
# ----------------------------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------------------------

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[interviewer, potential_customer, ux_converger],
  tasks=[interview, answer_questions, create_challenge],
  process=Process.sequential  # Optional: Sequential task execution is default
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'Doing laundry'})
print(result)