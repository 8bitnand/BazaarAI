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
    """
    Given the company name get scrip code.
    This function looks up the copmany name from a json map 'company_name': 'scrip_code' and return the scrip_code
    inputs:
        company_name: str
    outputs:
        company_scrip code: str
    """

    best, score, _ = process.extractOne(
        company_name, list(norm_to_orig), scorer=fuzz.token_sort_ratio
    )
    if score <= 50:
        raise ToolException(f"Error: cannot find company with name {company_name}.")

    return int(stk[best])


async def async_get_company_scrip_code(company_name: str) -> str:
    """
    Given the company name get scrip code.
    This function looks up the copmany name from a json map 'company_name': 'scrip_code' and return the scrip_code
    inputs:
        company_name: str
    outputs:
        company_scrip code: str
    """
    best, score, _ = process.extractOne(
        company_name, list(norm_to_orig), scorer=fuzz.token_sort_ratio
    )
    if score <= 50:
        raise ToolException(f"Error: cannot find company with name {company_name}.")

    return int(stk[best])


def get_stock_quote(scrip_code: str) -> dict:
    """
    Given the scrip code get company details / quote.
    This function get the json detail about the company, returns a dict/ json
    inputs:
        scrip_code: str
    outputs:
        company_quote: dict / json
    """
    return bse.getQuote(str(scrip_code))


async def async_get_stock_quote(scrip_code: str) -> dict:
    """
    Given the scrip code get company details / quote.
    This function get the json detail about the company, returns a dict / json
    inputs:
        scrip_code: str
    outputs:
        company_quote: dict / json
    """
    return bse.getQuote(str(scrip_code))


#
def get_top_losers() -> dict:
    """
    Gets the List of all top losers of the day. return a list of dict
    inputs:
        _
    outputs:
        top_losers: list[dict]
    """
    return bse.topLosers()


async def async_get_top_losers() -> dict:
    """
    Gets the List of all top losers of the day. return a list of dict
    inputs:
        _
    outputs:
        top_losers: list[dict]
    """
    return bse.topLosers()


def get_top_gainers() -> dict:
    """
    Gets the List of all top gainers of the day. return a list of dict
    inputs:
        _
    outputs:
        top_lgainers: list[dict]
    """
    return bse.topGainers()


async def async_get_top_gainers() -> dict:
    """
    Gets the List of all top gainers of the day. return a list of dict
    inputs:
        _
    outputs:
        top_gainers: list[dict]
    """
    return bse.topGainers()


company_scrip_code = StructuredTool.from_function(
    func=get_company_scrip_code,
    coroutine=async_get_company_scrip_code,
    name="get_company_scrip_code",  # ENSURE this matches your prompt
    description="Get scrip code from company name",  # Helps LLM select it
)

company_details = StructuredTool.from_function(
    func=get_stock_quote,
    coroutine=async_get_stock_quote,
    name="get_stock_quote",
    description="Get company details from scrip code",
)


top_losers = StructuredTool.from_function(
    func=get_top_losers,
    coroutine=async_get_top_losers,
    name="get_top_losers",
    description="Get top losers of the day",
)

top_gainers = StructuredTool.from_function(
    func=get_top_gainers,
    coroutine=async_get_top_gainers,
    name="get_top_gainers",
    description="Get top losers of the day",
)
