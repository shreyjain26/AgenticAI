import os
import asyncio
import tempfile
from playwright.async_api import async_playwright
import autogen
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen import register_function

llm_config = {"model":"gpt-4o",
        "api_key": "api-key",
        "base_url": "baseurl",
        "api_type": "azure",
        "api_version": "2023-03-15-preview"
    }
 
async def take_screenshot(url: str):
    async with async_playwright() as p:

        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        await page.screenshot(path=temp_file.name)
        await browser.close()
        print("Screenshot taken at")
        return temp_file.name

 
Image_explainer = MultimodalConversableAgent(
    name="image-explainer-1",
    max_consecutive_auto_reply=10,
    llm_config=llm_config,
    system_message='''Act as an image description generator. Describe the given image in terms of its login page and forms, focusing on placeholders and buttons. so that coder will write the python playwright code''',
)

 
coder = autogen.AssistantAgent(
    name="Coder",
    system_message="""You follow an Image_explainer. You write python Playwright code for the same which explained by image_explainer for login page. if you need any credintials like user name and password ask to user_proxy. and hit sign in after receiving credintials.
    Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. 
    Don't use a code block if it's not intended to be executed by the executor. Don't include multiple code blocks in one response. 
    Do not ask others to copy and paste the result. Check the execution result returned by the executor. If the result indicates there is an error, fix the error and output the code again. 
    Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.""",
    llm_config=llm_config,
)
 
executor = autogen.AssistantAgent(
    name="Executer",
    system_message="""Execute code produced by the coder in the groupchat folder""",
    code_execution_config={"last_n_messages": 8, "work_dir": "groupchat", "use_docker": False},
    llm_config=llm_config,
)

url_identifier = autogen.AssistantAgent(
    name="UrlIdentifier",
    system_message="""Identify the URL from the user's prompt. Respond only with the URL. If no URL is found, you need to find the domain specific url as per prompt.""",
    llm_config=llm_config,
)
 
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="you should fallow the coder and check what its expecting from you and give the credintials like email and password when coder ask you to give",
    human_input_mode="ALWAYS",  # Try between ALWAYS, NEVER, and TERMINATE
    max_consecutive_auto_reply=10,
    code_execution_config={
        "use_docker": False
    }, 
)
 
url_identifier.register_for_llm(name="take_screenshot", description="A simple screenshot taker")(take_screenshot)
Image_explainer.register_for_execution(name="take_screenshot")(take_screenshot)
register_function(
    take_screenshot,
    caller=url_identifier,
    executor=Image_explainer,
    name="take_screenshot",
    description="A simple screenshot taker"
)

# We set max_round to 10
groupchat = autogen.GroupChat(agents=[url_identifier,Image_explainer,coder,executor, user_proxy], messages=[], max_round=10, speaker_selection_method='round_robin')

vision_capability = VisionCapability(lmm_config=llm_config)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
vision_capability.add_to_agent(group_chat_manager)

# prompt = "Describe the image of linkedin home page into appropriate python playwright code."

# prompt = "Explain everything there is in the web page of nvidia omviverse into appropriate playwright code."
prompt = "Login the linkedin home page and search for ai/ml internships by using appropriate playwright code."
# prompt = " I want you to show me how to use Azure Devops to deploy containers"
url_identified = url_identifier.generate_reply(messages=[{"content":prompt, "role":"user"}])
print(url_identified)

rst = user_proxy.initiate_chat(
    group_chat_manager,
    # message="""Describe the linkedin home page into appropriate python playwright code""",
                        #  https://www.linkedin.com/home""",
    message=prompt,
)
