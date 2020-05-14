import logging
import os

import honeycomb
from gqlpycgen.utils import now, json_dumps


HONEYCOMB_URI = os.getenv("HONEYCOMB_URI", "https://honeycomb.api.wildflower-tech.org/graphql")
HONEYCOMB_TOKEN_URI = os.getenv("HONEYCOMB_TOKEN_URI", "https://wildflowerschools.auth0.com/oauth/token")
HONEYCOMB_AUDIENCE = os.getenv("HONEYCOMB_AUDIENCE", "https://honeycomb.api.wildflowerschools.org")
HONEYCOMB_CLIENT_ID = os.getenv("HONEYCOMB_CLIENT_ID")
HONEYCOMB_CLIENT_SECRET = os.getenv("HONEYCOMB_CLIENT_SECRET")

def get_client():
    return honeycomb.HoneycombClient(
        uri=HONEYCOMB_URI,
        client_credentials={
            'token_uri': HONEYCOMB_TOKEN_URI,
            'audience': HONEYCOMB_AUDIENCE,
            'client_id': HONEYCOMB_CLIENT_ID,
            'client_secret': HONEYCOMB_CLIENT_SECRET,
        }
    )


def create_inference_execution(assignment_id, start, end, sources=list(), model="alphapose", version="v1", honeycomb_client=None):
    if honeycomb_client is None:
        honeycomb_client = get_client()
    query_pages = """
        mutation createInferenceExecution($inferenceExecution: InferenceExecutionInput) {
          createInferenceExecution(inferenceExecution: $inferenceExecution) { inference_id }
        }
        """
    variables = {
        "inferenceExecution": {
            "name": f"{assignment_id}::{start}-->>{end}",
            "notes": f"created by inference_helper in prepare job",
            "model": model,
            "version": version,
            "data_sources": sources,
            "execution_start": now(),
        }
    }
    result = honeycomb_client.raw_query(query_pages, variables)
    return result.get("createInferenceExecution").get("inference_id")
