from bsedata.bse import BSE
from langchain.tools import BaseTool, StructuredTool
from langchain_core.tools import ToolException
from typing import Any, Dict, List, Tuple
import json
from rapidfuzz import process, fuzz

# bse = BSE(update_codes=True)
bse = BSE()
DB_PATH = "src/db/stk_codes.json"
with open(DB_PATH, "r") as f:
    stk = json.load(f)
norm_to_orig = stk.keys()


def get_company_scrip_code(company_name: str) -> str:
    """Get scrip code from company name"""

    best, score, _ = process.extractOne(
        company_name, list(norm_to_orig), scorer=fuzz.token_sort_ratio
    )
    if score <= 50:
        raise ToolException(f"Error: cannot find company with name {company_name}.")

    return stk[best]


async def async_get_company_scrip_code(company_name: str) -> str:
    """Get scrip code from company name."""
    best, score, _ = process.extractOne(
        company_name, list(norm_to_orig), scorer=fuzz.token_sort_ratio
    )
    if score <= 50:
        raise ToolException(f"Error: cannot find company with name {company_name}.")

    return stk[best]


company_details = StructuredTool.from_function(
    func=get_company_scrip_code, coroutine=async_get_company_scrip_code
)
