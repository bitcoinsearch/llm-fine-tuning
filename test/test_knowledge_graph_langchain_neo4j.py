from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import en_core_web_lg
import os
import openai
from dotenv import load_dotenv
import warnings
from neo4j import GraphDatabase

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

nlp = en_core_web_lg.load()

# initialize database connection
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "psql1234"))

# # Neo4j AuraDB
# graph_aura_db = Neo4jGraph(
#     url="neo4j+s://75a805c6.databases.neo4j.io:7473",
#     username="neo4j",
#     password="<your-password>"
#     )

# Neo4j Desktop
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="psql1234"
)

# Sample Data
dataset = [
    {
        "Title": "Practical PTLCs, a little more concretely",
        "Summary": """The email discusses a downside to using a MuSig API over a single key. The downside is that new people may question the use of the MuSig API and someone knowledgeable will have to explain the historical context and the specific APIs used in libsecp256k1 at that time. The downside mentioned is related to both performance and potential confusion among new users."""
    },
    {
        "Title": "Payment Splitting; Switching and its impact on Balance Discovery Attacks (preprint)",
        "Summary": """Gijs van Dam begins the email by mentioning their previous discussion on Link MPP over alternative routes. They have developed a proof of concept called Payment Splitting &amp; Switching (PSS) by creating a core-lightning plugin. The plugin works by splitting up a payment into multiple parts, with one part following the original route and using the original onion but committing to an HTLC with a lesser amount. Upon receiving this HTLC, the recipient can forward the payment but will wait for another payment from the sender. The sender then sends this additional payment over an alternative route, contingent on the payment hash of the original payment.The interest in Link MPP and PSS arises from Gijs van Dam's research on the Balance Discovery Attack (BDA), also known as the probing attack. PSS allows for route changes and amount changes without the sender's knowledge, which impacts how an attacker interprets information obtained through a BDA. In an LN simulator using real-world data, deploying PSS resulted in up to a 62% drop in information gain compared to earlier work without PSS. Further details on this research can be found in Gijs van Dam's preprint on the Cryptology ePrint Archive.Gijs van Dam welcomes feedback on their research and would like to hear thoughts on the role PSS could play in mitigating probing and/or jamming. They have also written a set of blog posts introducing the research, which can be found at the provided links. Finally, they mention that PSS may also work with PTLC, but this will be discussed at another time.Overall, the email discusses Gijs van Dam's development of the Payment Splitting &amp; Switching (PSS) plugin, its potential applications in mitigating probing and/or jamming, and the impact of PSS on the Balance Discovery Attack (BDA). They provide links to additional resources for further reading and welcome feedback on their research."""
    },
    {
        "Title": "Scaling Lightning With Simple Covenants",
        "Summary": """In an email sent by ZmnSCPxj to the recipient, Dave, the former refers to an alternative that is depicted in a diagram on page 6 and described in the text on page 7 of a paper. The email does not provide further details or context regarding this alternative.The email begins with a greeting - "Good morning, ZmnSCPxj" - followed by a statement indicating that the alternative being referred to is what is shown in the aforementioned diagram and described in the accompanying text. However, the exact nature or purpose of this alternative is not explicitly mentioned in the email.It is important to note that the email does not include any links or additional information that could provide further context or explanation. The tone of the email remains formal throughout, and there are no indications of any specific questions or requests from either party.Overall, the email appears to be a brief communication between ZmnSCPxj and Dave regarding the presence of an alternative in a paper, without delving into the specifics or implications of this alternative."""
    },
    {
        "Title": "Bitcoin Research Day 2023",
        "Summary": """The email announces the upcoming Bitcoin Research Day, which will take place at Chaincode in Midtown NYC on October 27th, 2023. The purpose of the event is to bring together researchers and developers to discuss the robustness, security, and decentralization of the Bitcoin system. The day will consist of both longer format talks, featuring speakers such as Benedikt Bünz, Ethan Heilman, Ittay Eyal, Carla Kirk-Cohen, Murch, and Martin Zumsande, among others, as well as lightning talks that will give researchers and developers the opportunity to present ongoing research and development projects in approximately five minutes.Slots for the lightning talks are still available, and interested individuals can sign up through the provided link (https://www.brd23.com/lightning-talks). The email encourages participation from both the bitcoin-dev and lightning-dev communities and invites anyone interested to attend and contribute to the discussion. The event is an in-person gathering, and attendees are required to RSVP through the website (https://www.brd23.com/).In summary, the email announces the Bitcoin Research Day, a conference that aims to bring together researchers and developers to discuss the robustness, security, and decentralization of the Bitcoin system. The event will feature longer format talks by notable speakers and lightning talks for shorter presentations of ongoing research and development projects. Interested individuals are encouraged to sign up for lightning talk slots and RSVP for the in-person event through the provided links."""
    },
    {
        "Title": "Sidepools For Improving Payment Reliability At Scale",
        "Summary": """In this email, the sender discusses the difficulty of predicting the future in terms of allocating liquidity on the Lightning Network. They compare allocating liquidity to investing in stocks and highlight the challenges of accurately predicting payment influxes and avoiding over or under-allocation.To address the issue of mis-allocating funds, the sender proposes the concept of sidepools. Sidepools are parallel constructions that allow forwarding node operators to manage the allocation of funds without closing channels. Instead of hosting channels within the sidepool mechanism, actual Lightning Network channels remain on-chain. The sidepool acts as a service for HTLC-swapping to facilitate the allocation of funds in the channels.The sender emphasizes the benefits of retaining 2-participant channels instead of expanding to channels with more participants. This approach reduces the number of participants who know about every payment, thus preserving privacy. Additionally, they mention that sidepools can be implemented using Decker-Wattenhofer decrementing `nSequence` mechanisms without any changes to Bitcoin.The motivation for using sidepools lies in their ability to help maintain liquidity in existing channels. By relying on other participants in the sidepool to receive funds, forwarding node operators can avoid depleting channels. This approach reduces costs compared to opening new channels or performing onchain/offchain swaps. The sender also acknowledges that past performance is not indicative of future performance, highlighting the potential risks of doubling down on specific counterparties.Another advantage of sidepools is their potential to support scaling by mitigating liquidity fluctuations during large buyer movements. The sender suggests the implementation of channel factories for end users and sidepools for forwarding nodes to further enhance scalability.Overall, the sender proposes the use of sidepools as a solution to the challenges of allocating liquidity on the Lightning Network. They provide insights into the benefits of this approach and how it can be implemented while maintaining compatibility with existing Lightning Network design."""
    }
]


def add_to_graph(driver, summary):
    """Script to add data to the knowledge graph"""
    print(f"adding data to the graph...")
    with driver.session() as session:
        doc = nlp(summary)
        for sentence in doc.sents:
            for entity in sentence.ents:
                print(f"Entity: {entity.text} :: Sentence: {sentence.text}")
                session.run("MERGE (a:Entity {name: $entity}) "
                            "MERGE (b:Sentence {text: $sentence}) "
                            "MERGE (a)-[:APPEARS_IN]->(b)",
                            entity=entity.text, sentence=sentence.text)


if __name__ == "__main__":

    '''
    ## use below script to add data to the knowledge graph
    # add create a knowledge graph using the data
    for data in dataset:
        print("--")
        add_to_graph(driver, data["Title"] + " " + data["Summary"])
        '''

    # query the knowledge graph using Langchain
    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0), graph=graph, verbose=True
    )

    question = "What is the concept of Sidepools?"  # "What is the downside to using a MuSig API over a single key?"

    result = chain.run(question)
    print(f"Result: {result}")
    '''
    Result: 
    The concept of Sidepools refers to parallel constructions that enable forwarding node 
    operators to manage the allocation of funds without closing channels. This allows for improved payment 
    reliability at scale on the Lightning Network, as it addresses the challenge of predicting future liquidity 
    allocation.
    '''
