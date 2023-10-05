import en_core_web_sm
from neo4j import GraphDatabase

# initialize database connection
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "psql1234"))
# llm-ft-db, psql1234

# initialize nlp model
nlp = en_core_web_sm.load()


def get_db_info(driver):
    with driver.session() as session:
        info = session.run("CALL db.info()").single().value()
        print(info)


# creating knowledge graph
def add_to_graph(summary, driver):
    print(f"adding data to the graph...")
    with driver.session() as session:
        # process the summary to get entities
        doc = nlp(summary)

        # iterate over sentences in the summary
        for sentence in doc.sents:
            # iterate over entities in the sentence
            for entity in sentence.ents:
                # use entity as a node in the graph and sentence as a relationship
                session.run("MERGE (a:Entity {name: $entity}) "
                            "MERGE (b:Sentence {text: $sentence}) "
                            "MERGE (a)-[:APPEARS_IN]->(b)",
                            entity=entity.text, sentence=sentence.text)
                print(entity.text, sentence.text)


# query knowledge graph
def query_graph(driver, entity_name):
    print(f"querying the graph...")
    with driver.session() as session:
        result = session.run("MATCH (a:Entity)-[:APPEARS_IN]->(b:Sentence) "
                             "WHERE a.name = $entity_name "
                             "RETURN b.text as sentence",
                             entity_name=entity_name)

        for record in result:
            print(f"result: {record['sentence']}")


summaries = [
    """We are proceeding with the plan discussed at the summit to conduct a "dry run" of HTLC endorsement and local reputation tracking. The objectives of this plan are to validate the behavior of local reputation algorithms using real-world data, gather liquidity and slot utilization data for resource bucketing, and establish a common data export format for analysis.The plan consists of several phases. Firstly, we will collect anonymized forwarding data in a CSV file format. This format includes fields such as version, channel IDs, peer information, fee offered by the HTLC, outgoing liquidity and slots, timestamps, and settlement status. Certain fields marked with [P] must be randomized if exported to researching teams.Next, we will propagate an experimental TLV called "endorsement" through the network. This TLV will be included in the `update_add_htlc` message and will indicate whether the HTLC is endorsed or not. Forwarding nodes will propagate the same value they receive or set it to 0 if not present.Finally, we will implement local reputation algorithms and actively set the value of the `endorsed` TLV for outgoing HTLCs. This signal will be used solely for data collection purposes. It is suggested that senders choose a probability (default: 20%) to set `endorsed=1` for their payments.Throughout this process, we prioritize privacy and ensure that sensitive information is protected. We encourage node operators to participate in data collection and welcome any questions or assistance. For more details and references, please refer to the email.""",
    """The sender of the email clarifies their goal for their previous comments, stating that monetary-based denial of service (DoS) deterrence is still a valuable area of research. They express concern that others may have misinterpreted the summit notes to mean that monetary approaches are not worth exploring. The sender emphasizes that they provided a rough proof-of-work scheme as an example and not something to actively pursue. They discuss the requirement for using the chain tip as a clock or setting up local clocks with each peer to handle any synchronization issues. They explain that the suggested scheme involves commitment transaction updates and does not require message proof. They differentiate between reputation-based and monetary-based approaches, noting that proving things is more relevant in reputation-based approaches. The sender suggests ways to avoid situations involving high fees and distant timeouts. They explain the concept of getting paid for holding liquidity hostage and clarify the purpose of force closing a channel. They also mention the link between on-chain fees and the scheme they outlined. The sender speculates about the possibility of experimenting with these ideas on a platform that supports smaller Bitcoin denominations. They suggest Liquid as a potential option but acknowledge the challenges of setting it up.""",
    """The lack of real-world datasets for conducting simulations and experiments on Lightning can be limiting. However, collecting the proposed fields over a long period may potentially lead to re-identification of anonymized channel counterparties based on heuristics correlated with public graph data. Combining datasets from multiple collection points could further allow drawing conclusions on transferred amounts, channel liquidities, and even the payment destination's identity. Trust in the researchers is crucial as surrendering this data requires it. It is recommended to clarify upfront the time-boxed collection period, data storage, and access permissions. Defining the collection period is necessary to avoid incentivizing node operators to store HTLC data long-term.""",
    """The email discusses the importance of privacy and data handling in a research project. The ideal collection period for the data is limited to 6 months. The aim is to provide node operators with local tooling so that they don't need to export the data. If people are comfortable sharing their data, it will be handled following best practices and not shared further. The fields will be anonymized as mentioned in the original email. In response to concerns about timestamps, they can be fuzzed as only the resolution period matters. The sender believes that conducting research based on real-world data is challenging but worthwhile""",
    """Vincent expresses his lack of concern regarding privacy issues for the selected node running the dry run. He states that he is not buying anything significant with his research node, so there is no real privacy threat. The only potential privacy leak he identifies is certain fields, but they are irrelevant from an analysis perspective and can be faked using core lightning. He emphasizes the importance of ensuring that the fake channel ID remains constant to achieve 100% reproducibility and allow more people to verify and identify any mistakes. Vincent worries about being confined to a limited data set from real nodes and real bitcoin transactions, which may result in a lack of certainty. He questions if there is something he is missing regarding faking the channel ID and node pub key. Vincent also mentions that he has witnessed PhD programs failing to start due to a lack of real data examples""",
]

if __name__ == "__main__":
    get_db_info(driver)

    '''
    # add the summaries to the graph
    for summary in summaries:
        add_to_graph(summary, driver)
    '''

    # query the graph
    query_graph(driver, "HTLC")

    '''
    # Respone of above query:
    querying the graph...
    result: Defining the collection period is necessary to avoid incentivizing node operators to store HTLC data long-term.
    result: This TLV will be included in the `update_add_htlc` message and will indicate whether the HTLC is endorsed or not.
    result: This format includes fields such as version, channel IDs, peer information, fee offered by the HTLC, outgoing liquidity and slots, timestamps, and settlement status.
    result: We are proceeding with the plan discussed at the summit to conduct a "dry run" of HTLC endorsement and local reputation tracking.
    '''

