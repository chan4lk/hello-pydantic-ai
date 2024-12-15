from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

class BusinessTermSheet(BaseModel):
    fund_name: Optional[str] = Field(
        None,
        alias="Fund Name",
        description="The formal name of the Fund. This may include:\n"
            "- Specific fund types or structures (e.g., 'Growth Equity Fund', 'Global Real Estate Fund').\n"
            "- Names indicating strategies or themes (e.g., 'Tech Innovation Fund', 'Emerging Markets Fund').\n"
            "- Use of acronyms or abbreviations (e.g., 'ABC Fund', 'XYZ Growth')."
    )

    asset_class: Optional[str] = Field(
        None,
        alias="Asset Class",
        description="The type of asset being considered, such as private debt, private equity, venture capital, microfinance, infrastructure, real estate, or public equities."
    )
    
    investment_team_info: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Team",
        description="Information about the investment team, including the team's size, experience, and expertise in the asset class."
    )
    #Firm
    firm_manager: Optional[str] = Field(
        None,
        alias="Firm Manager",
        description="Investment Company Name. It often includes specific keywords or patterns such as:\n"
            "- Common Company Suffixes: '(Pvt) Ltd.', 'LLC', 'Inc.', 'Limited', 'Corporation', 'Co.', 'Corp.', etc.\n"
            "- Keywords and Formats: Multi-word names, names with punctuation (e.g., 'ABC (Pvt) Ltd.'), or acronyms like 'XYZ LLC'.\n"
            "- Names with Country (e.g., 'SEED DESIGN USA')."
    )
    
    website: Optional[str] = Field(
        None,
        alias="Website",
        description="The official website of the business or fund. It should:"
            "- Be a valid URL (e.g., 'https://www.companyname.com', 'www.companyname.com')."
            "- Provide detailed information about the business, fund, or its services."
            "- Often include pages like 'About Us', 'Investors', or 'Contact' for easy navigation."
            "- Use secure protocols like HTTPS wherever possible."
    )
    
    contact_name: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Contact Name",
        description="The contact information for inquiries or further details, Sales or Investor Relations Name."
    )

    job_title: Optional[str] = Field(
        None,
        alias="Job Title",
        description="Sales or Investor Relations Position, role"
    )

    contact_email: Optional[str] = Field(
        None,
        alias="Contact Email",
        description="The email address for Sales or Investor Relations inquiries. It must:"
            "- Be a valid email format (e.g., 'name@example.com')."
            "- Represent a specific department or individual responsible for fund communication."
            "- Common examples include 'hello@example.com' and 'info@example.com', often used for general inquiries."
            "- Often include keywords like 'sales', 'investor', or 'info' (e.g., 'sales@fundcompany.com')."
    )

    contact_number: Optional[str] = Field(
        None,
        alias="Contact Number",
        description="The primary phone or contact number for Sales or Investor Relations."
    )

    #Strategy
    fund_series: Optional[str] = Field(
        None,
        alias="Fund Series",
        description="Is this the first, second, third, fourth fund + for the firm?"
    )

    fund_close_date: Optional[str] = Field(
        None,
        alias="Fund Close Date",
        description="Final Close date"
    )

    target_fund: Optional[str] = Field(
        None,
        alias="Target Fund Size",
        description="The target size of the fund, typically measured in millions or billions of currency (e.g., USD)."
    )

    investment_ticket_size: Optional[str] = Field(
        None,
        alias="Investment Ticket Minimum Size",
        description="Minimum investment"
    )

    target_return: Optional[str] = Field(
        None,
        alias="Target Return",
        description="What return is the fund hoping to achieve? IRR / MOIC/ XX over benchmark outperformance"
    )

    carried_interest_fee: Optional[str] = Field(
        None,
        alias="Carried Interest Fee",
        description="Performance fee"
    )

    management_fee: Optional[str] = Field(
        None,
        alias="Management Fee",
        description="Investment Management Fee"
    )

    region: Optional[str] = Field(
        None,
        alias="Region",
        description="The primary geographic region of investment, specifying the area where the organization is currently focused. "
        "This can include regions such as specific countries, continents, or market zones (e.g., North America, APAC, Europe). "
        "Optionally, it can also outline potential future expansion areas or emerging markets being considered for investment."
    )

    fund_inception: Optional[str] = Field(
        None,
        alias="Fund Inception Year",
        description="What year was did the fund open?"
    )

    liquidity: Optional[str] = Field(
        None,
        alias="Liquidity",
        description="Does the fund offer monthly or quarterly or daily liquidity?"
    )

    vintage_year: Optional[str] = Field(
        None,
        alias="Vintage Year",
        description="Which year will the fund be launched?"
    )

    status: Optional[str] = Field(
        None,
        alias="Status",
        description="Evergreen, liquidated, coming to market, closed-end, closed fund"
    )

    team_tr: Optional[str] = Field(
        None,
        alias="Team TR",
        description="Years experience in this asset class from the lead PM?"
    )

    term: Optional[str] = Field(
        None,
        alias="Term",
        description="The length of the investment or fund's holding period, expressed in years (e.g., 3-5 years, 10-12 years)."
    )

    key_financial_data: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Key Financial Data",
        description="Key financial metrics or historical data points, such as 1, 3, and 5-year returns, from prior funds or investments managed by the firm. 1, 3, 5 year return numbers (prior Track Record from other fund or prior fund with other company): "
    )

    strategy_differentiator: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Strategy Differentiator",
        description="Defines the unique edge of the strategy. Key aspects may include:\n"
            "- Team expertise and specialized skills."
            "- Innovative deal sourcing techniques."
            "- Due diligence (DD) methodologies and thoroughness."
            "- Advanced underwriting experience."
            "- Strong execution capabilities."
            "- Robust servicing and operational excellence."
            "Accepts either a string or a structured dictionary format for detailed descriptions."
            "Operational Efficiency,Product Accessibility,Flexible Logistics"
    )

    number_of_holdings: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Number of holdings",
        description="How many investments in the portfolio? Can be a range?"
    )

    un_sustainable_development_goal: Optional[str] = Field(
        None,
        alias="UN Sustainable Development Goal",
        description=(
            "The specific SDG the business aligns with. "
            "Options include: No poverty (SDG 1), Zero hunger (SDG 2), Good health and well-being (SDG 3), "
            "Quality education (SDG 4), Gender equality (SDG 5), Clean water and sanitation (SDG 6), "
            "Affordable and clean energy (SDG 7), Decent work and economic growth (SDG 8), Industry, innovation and infrastructure (SDG 9), "
            "Reduced inequalities (SDG 10), Sustainable cities and communities (SDG 11), Responsible consumption and production (SDG 12), "
            "Climate action (SDG 13), Life below water (SDG 14), Life on land (SDG 15), Peace, justice, and strong institutions (SDG 16), "
            "and Partnerships for the goals (SDG 17). If not specified, return 'Not Mentioned'."
        )
    )

    impact_framework: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Impact Framework Used",
        description="Internal or external"
    )

    impact_hutdle: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Impact Hurdle",
        description="Is there a specific goal mentioned around level of impact aiming to achieve?"
    )

    impact_intergration_method: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Impact Integration Method",
        description="How is impact integrated into the investment process. Is impact integral to the investment selection, due diligence process, exit considerations for the investment?"
    )

    formalized_impact_team: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Formalized Impact Team",
        description="Details about the organization's internal resources focused on impact-related initiatives."
    )

    type_impact_metric: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Type of Impact Metric",
        description="Environmental or Social"
    )

    outcomes_mesured: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="Outcomes Measured",
        description="Details the outcomes targeted by the fund, aligned with environmental, social, and governance (ESG) objectives."
    )

@dataclass
class ExtractorDependencies:
    text: str

class BusinessTermExtractor(Agent[ExtractorDependencies, BusinessTermSheet]):
    def __init__(self):
        super().__init__(
            'openai:gpt-4',
            deps_type=ExtractorDependencies,
            result_type=BusinessTermSheet,
            system_prompt=(
                'You are an expert financial analyst specializing in extracting business terms from pitch decks. '
                'Analyze the provided text carefully and extract all relevant business terms. '
                'For any field where the information is not explicitly mentioned in the pitch deck, '
                'use the exact string "Not mentioned" as the value. Do not make assumptions or use '
                'placeholder values. If a value is unclear or ambiguous, use "Not mentioned".'
            )
        )

    def extract(self, text: str) -> BusinessTermSheet:
        deps = ExtractorDependencies(text=text)
        result = self.run_sync('Extract business terms from the provided text. Use "Not mentioned" for any information not explicitly found.', deps=deps)
        return BusinessTermExtractor.validate_result(result.data)

    @staticmethod
    def validate_result(terms: BusinessTermSheet) -> BusinessTermSheet:
        """Ensure all None or empty values are set to 'Not mentioned'"""
        data = terms.model_dump()
        for key, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                data[key] = "Not mentioned"
            elif isinstance(value, dict):
                # Handle dictionary values
                if not value:  # Empty dict
                    data[key] = "Not mentioned"
            elif isinstance(value, list):
                # Handle list values
                if not value:  # Empty list
                    data[key] = "Not mentioned"
        return BusinessTermSheet(**data)

@dataclass
class ReviewerDependencies:
    original_text: str
    current_terms: str
    missing_fields: list[str]

class BusinessTermReviewer(Agent[ReviewerDependencies, BusinessTermSheet]):
    def __init__(self):
        super().__init__(
            'openai:gpt-4',
            deps_type=ReviewerDependencies,
            result_type=BusinessTermSheet,
            system_prompt=(
                'You are an expert financial analyst reviewer. Your task is to review the extracted business terms '
                'and identify any missing information that should be present in the pitch deck. '
                'For any field where the information cannot be found in the pitch deck, '
                'use the exact string "Not mentioned". Do not make assumptions or use placeholder values.'
            )
        )

    def review(self, terms: BusinessTermSheet, text: str) -> BusinessTermSheet:
        # Only review fields that are None or empty
        missing_fields = [
            field for field, value in terms.model_dump().items() 
            if value is None or value == "" or value == {} or value == []
        ]
        
        if not missing_fields:
            return terms

        deps = ReviewerDependencies(
            original_text=text,
            current_terms=terms.model_dump_json(indent=2),
            missing_fields=missing_fields
        )
        
        result = self.run_sync(
            'Review the current terms and find missing information for the specified fields. Use "Not mentioned" if information is not found.',
            deps=deps
        )
        
        # Merge the new results with existing non-null values
        updated_terms = BusinessTermSheet(**{
            **terms.model_dump(),
            **{k: v for k, v in result.data.model_dump().items() if v is not None and v != ""}
        })
        
        return BusinessTermReviewer.validate_result(updated_terms)

    @staticmethod
    def validate_result(terms: BusinessTermSheet) -> BusinessTermSheet:
        """Ensure all None or empty values are set to 'Not mentioned'"""
        data = terms.model_dump()
        for key, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                data[key] = "Not mentioned"
            elif isinstance(value, dict):
                if not value:  # Empty dict
                    data[key] = "Not mentioned"
            elif isinstance(value, list):
                if not value:  # Empty list
                    data[key] = "Not mentioned"
        return BusinessTermSheet(**data)

class BusinessTermAdmin:
    def approve_terms(self, terms: BusinessTermSheet) -> tuple[bool, str]:
        """
        Simulate admin approval process.
        Returns a tuple of (approved: bool, feedback: str)
        """
        # Count fields that have actual content (not "Not mentioned")
        total_fields = len(terms.model_fields)
        filled_fields = len([
            v for v in terms.model_dump().values() 
            if v is not None and v != "Not mentioned"
        ])
        completion_rate = filled_fields / total_fields

        if completion_rate < 0.3:  # Lowered threshold since we're being more strict with "Not mentioned"
            return False, f"Only {filled_fields}/{total_fields} fields have actual content. Need more information."
        
        return True, f"Approved with {filled_fields}/{total_fields} fields containing actual content."
