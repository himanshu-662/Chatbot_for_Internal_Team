import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Reconfigure stdout to use UTF-8 just in case
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_pdf(filename, title, paragraphs, access_level):
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Setup document
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        rightMargin=54, 
        leftMargin=54,
        topMargin=54, 
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=6,
        textColor='#1A365D' # Sleek deep corporate blue
    )
    badge_style = ParagraphStyle(
        'AccessBadge',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=14,
        textColor='#718096' # Slate gray for metadata badge
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14.5,
        spaceAfter=9
    )
    
    story = [
        Paragraph(title, title_style),
        Paragraph(f"<b>CLASSIFICATION / ACCESS LEVEL:</b> {access_level.upper()}", badge_style),
        Spacer(1, 0.05 * inch)
    ]
    
    for p_text in paragraphs:
        story.append(Paragraph(p_text, body_style))
        
    doc.build(story)
    print(f"Successfully generated: {filename}")

def main():
    # Generate directly into the data folder used by RAG ingestion pipeline
    base_dir = "data"
    
    # Clean old documents in base_dir to avoid stale files
    if os.path.exists(base_dir):
        import shutil
        shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)

    
    docs_to_create = {
        # =========================================================================
        # 1. PUBLIC / ALL EMPLOYEES (Access: all_employees, engineering, hr, finance, management)
        # =========================================================================
        "public/company_handbook.pdf": (
            "DMS Solutions Employee Handbook & Office Guidelines",
            "Public / All Employees",
            [
                "Welcome to DMS Solutions. This handbook outlines our general operational rules, working hours, and workplace policies applicable to all team members globally.",
                "<b>Core Working Hours:</b> Standard business hours are 9:00 AM to 6:00 PM local office time, Monday through Friday. Core collaboration hours when all team members are expected to be reachable are 10:00 AM to 4:00 PM.",
                "<b>Hybrid Work Policy:</b> Employees in eligible roles may work remotely up to 2 days per week with manager coordination. Tuesdays and Thursdays are designated as in-office collaboration days.",
                "<b>Workplace Dress Code:</b> DMS Solutions maintains a Smart Casual dress code. When meeting clients or attending external partner conferences, business casual attire is expected.",
                "<b>Holiday Calendar 2026:</b> All regional offices observe official public holidays including New Year's Day (Jan 1), Memorial Day (May 25), Independence Day (Jul 3 observed), Labor Day (Sep 7), Thanksgiving (Nov 26), and Christmas Day (Dec 25). Each employee also receives 2 floating cultural holidays per year."
            ]
        ),
        "public/benefits_overview.pdf": (
            "General Employee Perks and Wellness Benefits",
            "Public / All Employees",
            [
                "DMS Solutions is committed to supporting every employee's health, wellbeing, and work-life balance through our comprehensive benefits program.",
                "<b>Paid Time Off (PTO):</b> All full-time employees accrue 20 days of paid vacation per calendar year, plus standard public holidays. Unused PTO up to 5 days can be rolled over to the following year.",
                "<b>Medical, Dental, and Vision:</b> Comprehensive health insurance coverage is provided to all full-time staff from their first day of employment. The company covers 85% of individual health insurance premiums.",
                "<b>Monthly Wellness Stipend:</b> Every employee is eligible for a $100 monthly wellness allowance that can be utilized for gym memberships, fitness trackers, yoga classes, or mental wellness subscriptions.",
                "<b>Annual Learning Budget:</b> To foster continuous personal growth, each team member is allocated $1,500 per year for approved technical books, online certifications, and educational workshops."
            ]
        ),

        # =========================================================================
        # 2. ENGINEERING (Access: engineering, management)
        # =========================================================================
        "engineering/system_architecture.pdf": (
            "Engineering Infrastructure & System Architecture Blueprint",
            "Engineering & Technical Operations",
            [
                "This technical blueprint describes the multi-region microservices infrastructure supporting the DMS Solutions Core Platform.",
                "<b>Cloud Infrastructure:</b> All production workloads run on Amazon Web Services (AWS) across two primary regions: us-east-1 (Primary) and us-west-2 (Disaster Recovery Failover). Compute is managed via Amazon Elastic Kubernetes Service (EKS).",
                "<b>Service Topology & Event Streaming:</b> Our backend consists of 45 containerized Go and Python microservices communicating asynchronously via an Apache Kafka distributed event bus with a 7-day retention period. Synchronous inter-service calls use gRPC with Protobuf schemas.",
                "<b>Databases & Caching Layer:</b> Transactional application data is persisted in Amazon Aurora PostgreSQL with multi-AZ replication. Redis Enterprise clusters provide sub-millisecond in-memory caching and distributed session management.",
                "<b>High Availability & SLA:</b> The infrastructure is designed for 99.99% uptime with automated health checks, multi-zone pod autoscaling, and cross-region database read-replicas."
            ]
        ),
        "engineering/deployment_and_ci_cd.pdf": (
            "CI/CD Deployment Pipelines and Release Protocols",
            "Engineering & DevOps",
            [
                "To maintain software quality and rapid delivery velocity, engineering teams follow strict continuous integration and deployment standards.",
                "<b>Git Branching Strategy:</b> We utilize a Trunk-Based Development workflow. Feature branches must be short-lived (less than 48 hours) and require automated linting, unit testing (minimum 80% code coverage), and security vulnerability scanning.",
                "<b>Pull Request Approvals:</b> Every Pull Request requires mandatory review and approval from at least two senior engineers before merging into the main branch.",
                "<b>Deployment Workflow & Staging:</b> Automated GitHub Actions build container images pushed to AWS ECR. Successful builds deploy automatically to the Dev and Staging clusters. Staging runs automated end-to-end integration tests nightly.",
                "<b>Production Releases & Rollbacks:</b> Production deployments use Blue-Green deployments via ArgoCD. Automated canary analysis monitors error rates (HTTP 5xx) for 15 minutes. If error rates exceed 0.5%, traffic automatically rolls back to the previous stable release."
            ]
        ),
        "engineering/internal_api_security.pdf": (
            "Internal API Security, Gateway Auth & Secrets Management",
            "Engineering & InfoSec",
            [
                "This document establishes the security guidelines for service-to-service communication, secret rotation, and API authentication.",
                "<b>API Gateway & Token Authentication:</b> External traffic is routed through Kong API Gateway. All incoming client requests must present a cryptographically signed OAuth2 / OIDC JWT token issued by our central identity provider.",
                "<b>Rate Limiting Rules:</b> Standard public API endpoints are throttled to 1,000 requests per minute per tenant. Burst allowances up to 2,500 requests per minute are granted to enterprise tiers.",
                "<b>Internal Service-to-Service Security:</b> All internal pod-to-pod communication is encrypted in transit using Mutual TLS (mTLS) enforced by Istio service mesh.",
                "<b>Secret Rotation Policy:</b> All API keys, database credentials, and symmetric encryption keys must be managed in HashiCorp Vault. Vault rotates production database credentials every 90 days automatically."
            ]
        ),

        # =========================================================================
        # 3. HUMAN RESOURCES (Access: hr, management)
        # =========================================================================
        "hr/salary_bands_and_compensation.pdf": (
            "Confidential Employee Compensation & Leveling Salary Bands",
            "HR & Talent Management Confidential",
            [
                "This confidential HR reference outlines the compensation structures, base salary ranges, and equity allocations for all standardized engineering and product job levels for the 2026 fiscal year.",
                "<b>Level 1 (Junior / Associate Engineer):</b> Base Salary Band: $75,000 - $95,000 USD. Annual performance bonus target: 5%. Standard equity grant: 2,500 RSUs over a 4-year vesting schedule.",
                "<b>Level 2 (Mid-Level Software Engineer):</b> Base Salary Band: $105,000 - $135,000 USD. Annual bonus target: 8%. Equity grant: 6,000 RSUs.",
                "<b>Level 3 (Senior Software Engineer):</b> Base Salary Band: $145,000 - $185,000 USD. Annual bonus target: 12%. Equity grant: 12,000 RSUs.",
                "<b>Level 4 (Staff Engineer / Engineering Manager):</b> Base Salary Band: $195,000 - $240,000 USD. Annual bonus target: 15%. Equity grant: 22,000 RSUs.",
                "<b>Level 5 (Principal Engineer / Director):</b> Base Salary Band: $260,000 - $320,000 USD. Annual bonus target: 20%. Equity grant: 40,000 RSUs with executive discretionary bonus.",
                "<b>Compensation Review Cycle:</b> Salary adjustments and merit increases take place annually in December, effective January 1st."
            ]
        ),
        "hr/performance_and_pip_guidelines.pdf": (
            "Performance Improvement Plan (PIP) & Corrective Action Protocols",
            "HR Operations Confidential",
            [
                "This guideline details the standardized Performance Improvement Plan (PIP) procedure and corrective management process for employees failing to meet performance standards.",
                "<b>Identification & Verbal Warning:</b> If an employee receives a performance rating below 'Meets Expectations' or demonstrates persistent delivery deficiencies, the direct manager must conduct a formal 1-on-1 discussion documented with HR.",
                "<b>Formal 30-Day PIP Process:</b> If performance does not improve within 30 days of the verbal warning, a formal 30-Day PIP document is initiated. The plan outlines specific, measurable milestones and weekly check-in evaluations.",
                "<b>Outcomes & Resolution:</b> Successful completion of all PIP milestones restores the employee to good standing. If the employee fails to meet the documented criteria after 30 days, HR and legal initiate mutual separation or employment termination with standard severance."
            ]
        ),
        "hr/internal_grievance_policy.pdf": (
            "Workplace Grievance, Harassment & Whistleblower Investigation Policy",
            "HR & Legal Confidential",
            [
                "DMS Solutions maintains a zero-tolerance policy against workplace harassment, discrimination, and unethical conduct. This document outlines investigation procedures.",
                "<b>Reporting Channels:</b> Employees may report grievances directly to their HR Business Partner, People Operations Lead, or anonymously via our confidential 24/7 whistleblower hotline (ethics@dmssolutions.internal).",
                "<b>Investigation Timeline:</b> All formal complaints are acknowledged within 24 hours. Formal investigations conducted by HR and internal legal counsel must conclude within 10 business days.",
                "<b>Anti-Retaliation Protection:</b> Strict anti-retaliation protections apply to all reporting individuals. Any disciplinary action against an employee for participating in a good-faith investigation is grounds for immediate dismissal."
            ]
        ),

        # =========================================================================
        # 4. FINANCE (Access: finance, management)
        # =========================================================================
        "finance/q3_departmental_budgets.pdf": (
            "Q3 2026 Approved Departmental Budget & Capital Allocation",
            "Finance & Accounting",
            [
                "This document records the official Q3 2026 fiscal budget allocations approved by the Financial Planning & Analysis (FP&A) team and the Chief Financial Officer.",
                "<b>Engineering Division:</b> Total Allocated: $4,800,000 USD. Breakdown: Cloud Infrastructure & AWS hosting ($2,200,000), Core Headcount & Contractors ($2,000,000), Software Tooling & SaaS Licenses ($600,000).",
                "<b>Sales & Marketing:</b> Total Allocated: $2,500,000 USD. Breakdown: Digital Enterprise Ads & Sponsorships ($1,400,000), Field Events & Conferences ($600,000), CRM & Lead Gen Tooling ($500,000).",
                "<b>Human Resources & Talent Acquisition:</b> Total Allocated: $1,100,000 USD. Breakdown: External Recruitment Agency Fees ($450,000), Employee Benefits Programs ($400,000), L&D and Wellness Stipends ($250,000).",
                "<b>Executive, Legal & General Operations:</b> Total Allocated: $1,600,000 USD. Breakdown: Office Leases & Facilities ($900,000), External Legal Counsel & Compliance Audits ($500,000), Administrative Services ($200,000).",
                "<b>Total Corporate Q3 Operating Budget:</b> $10,000,000 USD."
            ]
        ),
        "finance/expense_and_travel_policy.pdf": (
            "Corporate Expense Limits, Reimbursement & Travel Policy",
            "Finance & Corporate Card Management",
            [
                "All employees incurring company-approved travel and business expenses must strictly adhere to the spending caps and reimbursement rules specified below.",
                "<b>Air Travel Guidelines:</b> Domestic flights must be booked in Standard Economy class. International flights with single flight legs exceeding 6 hours are eligible for Premium Economy or Business Class with prior VP approval.",
                "<b>Lodging Nightly Limits:</b> Standard corporate hotel allowance is capped at $250 per night (inclusive of taxes). For high-cost designated tier-1 cities (New York, San Francisco, London, Tokyo), the nightly cap is $375.",
                "<b>Daily Meals & Per-Diem:</b> Meal expenses are reimbursed up to a maximum per-diem of $75 per day ($15 Breakfast, $25 Lunch, $35 Dinner). Itemized receipts are mandatory for all transactions above $25.",
                "<b>Submission Deadlines:</b> Expense reports must be submitted via Concur within 30 days of expense occurrence. Claims submitted after 60 days will be automatically rejected."
            ]
        ),
        "finance/vendor_contracts_and_procurement.pdf": (
            "Enterprise Vendor Contracts, Commitments & Procurement Matrix",
            "Finance & Procurement",
            [
                "This document records annual commitments for key enterprise vendor agreements and outlines the delegation of procurement approval authority.",
                "<b>Major Cloud & Software Commitments:</b>",
                "• Amazon Web Services (AWS 3-Year EDP): $1,400,000 annual spend commitment with an enterprise discount rate of 18%.",
                "• Salesforce Enterprise CRM: $380,000 annual contract covering 450 global sales seats.",
                "• Datadog Observability Suite: $180,000 annual contract for infrastructure metrics and APM logs.",
                "• Slack Enterprise Grid & Google Workspace: $150,000 combined annual licensing.",
                "<b>Procurement Approval Matrix:</b>",
                "• Contracts under $10,000: Direct Department Manager sign-off required.",
                "• Contracts $10,000 - $50,000: Department VP and FP&A Director approval required.",
                "• Contracts exceeding $50,000: CFO and Chief Legal Officer formal execution required."
            ]
        ),

        # =========================================================================
        # 5. MANAGEMENT / EXECUTIVE (Access: management)
        # =========================================================================
        "management/m_and_a_acquisition_strategy.pdf": (
            "CONFIDENTIAL: M&A Strategic Acquisition Plan - Project Titan",
            "Executive Board & Senior Management Eyes Only",
            [
                "STRICTLY CONFIDENTIAL: This strategic planning document outlines the corporate acquisition strategy codenamed 'Project Titan' for fiscal year 2026.",
                "<b>Target Company Profile:</b> CloudMetrics Inc., a Boston-based startup specializing in automated AI telemetry and distributed observability analytics with $6.2M ARR and 38 enterprise customers.",
                "<b>Proposed Transaction Terms:</b> Total acquisition valuation negotiated at $38,000,000 USD, structured as 65% cash ($24.7M) and 35% DMS Solutions Series C Preferred Stock ($13.3M).",
                "<b>Strategic Rational & Synergy:</b> Integrating CloudMetrics' proprietary machine learning engine into DMS Solutions' core platform will accelerate our AI monitoring roadmap by an estimated 18 months.",
                "<b>Execution Timeline & Milestones:</b> Non-binding Letter of Intent (LOI) signing targeted for October 15, 2026. Confidential confirmatory due diligence will run through November, with formal public closing announced on December 10, 2026."
            ]
        ),
        "management/board_meeting_minutes_2026.pdf": (
            "CONFIDENTIAL: Board of Directors Meeting Minutes & Resolutions",
            "Executive Board & Senior Management Eyes Only",
            [
                "CONFIDENTIAL RECORD: Minutes of the DMS Solutions Board of Directors Meeting held on July 14, 2026.",
                "<b>Executive Compensation & Bonus Targets:</b> The compensation committee approved the revised 2026 CEO and Executive Officer performance bonus metrics tied to achieving $95M in Annual Recurring Revenue (ARR) and 20% operating margin.",
                "<b>Equity Pool Refresh:</b> The board unanimously approved expanding the Employee Stock Option Pool by 3,500,000 shares (a 12% pool refresh) to support upcoming strategic senior hires in AI engineering.",
                "<b>European Subsidiary Expansion:</b> Resolution passed to establish a fully-owned operating entity in Frankfurt, Germany (DMS Solutions GmbH) by Q4 2026 to satisfy EU data sovereignty and GDPR compliance for banking clients."
            ]
        ),
        "management/executive_growth_roadmap.pdf": (
            "CONFIDENTIAL: Multi-Year Strategic Growth Plan & Series C Financing",
            "Executive Leadership Eyes Only",
            [
                "CONFIDENTIAL: This memorandum outlines leadership priorities for capital raising, business unit restructuring, and long-term valuation targets.",
                "<b>Series C Venture Financing:</b> DMS Solutions is preparing to initiate a $60,000,000 Series C funding round in Q1 2027 at a target pre-money valuation of $400,000,000 USD. Lead discussions are underway with Apex Venture Partners.",
                "<b>Organizational Restructuring:</b> Phasing out legacy on-premise professional consulting services by end of Q4 to refocus 100% of engineering bandwidth on the high-margin enterprise SaaS subscription product.",
                "<b>Revenue Goals:</b> Target ARR trajectory: $95M in 2026, $140M in 2027, and positioning the company for initial public offering (IPO) readiness by late 2028."
            ]
        )
    }
    
    print("Starting generation of Role-Based Mock Corporate PDF documents...")
    
    for relative_path, (title, access_level, paragraphs) in docs_to_create.items():
        full_path = os.path.join(base_dir, relative_path)
        create_pdf(full_path, title, paragraphs, access_level)
        
    print("\nGeneration complete! All mock corporate PDF documents created in 'data/' folder.")

if __name__ == "__main__":
    main()
