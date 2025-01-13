import os

def main():
    # Define templates for management companies.
    # We keep references to “Pinetree Country Club,” emphasizing multi-course or multi-property benefits.
    
    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a single-course solution into a platform that can streamline F&B operations across multiple properties under your management—handling on-course orders, snack-bar requests, and to-go pickups. Our goal is to boost operational efficiency without adding complexity for your teams.

We’re inviting 2–3 management companies to join us at no cost in 2025, ensuring our platform addresses your unique challenges. For instance, at Pinetree Country Club, we helped reduce average order times by 40%, leading to higher guest satisfaction and smoother golf rounds.

Interested in a quick chat about how this might work for [ManagementCompanyName] and its facilities? We’d love to share how Swoop can elevate efficiency and revenue across your portfolio.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has developed into a multi-facility F&B platform—covering on-course deliveries, streamlined snack-bar orders, and convenient to-go pickups. We help management companies maintain a high level of service while optimizing labor and keeping pace of play on track.

At Pinetree Country Club, we saw average order times drop by 40%, which boosted satisfaction and cut bottlenecks. Would you have a few minutes to see if [ManagementCompanyName]’s courses could see a similar benefit?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf supports management companies by acting as a centralized ordering solution—coordinating beverage cart requests, snack-bar orders, and to-go services across multiple properties. We designed the platform to improve operational flow and keep golfers engaged.

At Pinetree Country Club, our approach reduced average order times by 40%. I’d love to explore how we could replicate that success for [ManagementCompanyName] at scale. Would you be open to a 10-minute call next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is more than a single-course solution—it’s a scalable platform that can unify F&B operations across all courses in your portfolio. We handle beverage cart requests, snack-bar orders, and to-go pickups in one streamlined interface, reducing wait times and boosting revenue potential.

We’re looking for 2–3 management companies to partner with in 2025 at no cost. One of our current partners, Pinetree Country Club, saw a 54% boost in F&B revenue by making orders seamless during each round.

If you're open to a quick chat, I'd love to see how [ManagementCompanyName] could benefit in the same way across your operations. Let me know a good time to connect, and I can share references or more details tailored to your needs.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform accommodates multiple properties under your management—covering beverage cart deliveries, snack-bar orders, and to-go requests. By centralizing these services, we can help you reduce bottlenecks and streamline operations portfolio-wide.

One of our recent partners, Pinetree Country Club, saw a 40% decrease in average order times. Let’s schedule a short call to explore how Swoop could improve F&B efficiency across [ManagementCompanyName]’s courses. When might you be free next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is set to help [ManagementCompanyName] deliver a superior F&B experience across all the facilities you manage. By integrating on-course orders, snack-bar pickups, and to-go services into one platform, we reduce wait times and simplify workflows for your teams.

At Pinetree Country Club, we drove a 54% boost in F&B revenue—a success we believe can be replicated on multiple courses under your management. Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss this further? If another time suits you better, let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course ordering platform helps multi-course management companies streamline F&B operations. From beverage cart deliveries to snack-bar orders, our platform can be customized for each facility, maintaining consistent service standards across all properties.

We're inviting 2–3 management companies to partner with us at no cost for 2025. For example, at Pinetree Country Club, we helped increase F&B revenue by 54% and cut average wait times by 40%. We believe [ManagementCompanyName]’s courses could see similar outcomes.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? Let me know if another time is better.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf’s platform to help [ManagementCompanyName] unify on-course F&B operations across your portfolio. We consolidate beverage cart orders and snack-bar pickups in one system, reducing strain on staff and elevating the guest experience.

We recently assisted a facility that achieved a 54% jump in F&B revenue and a 40% drop in wait times—a transformation we believe is replicable at scale. 

Let’s set up a brief 10-minute conversation to see if our platform aligns with your management goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s streamlined platform can strengthen your entire portfolio—whether guests order at the turn, from a beverage cart, or via to-go pickups. By centralizing F&B orders, we help management companies reduce wait times and boost revenue throughout all managed facilities.

At Pinetree Country Club, our solution led to a 54% increase in F&B sales and a 40% reduction in wait times. Industry insights show that digital ordering often provides a 20–40% lift in ancillary revenue. It could be a major win for [ManagementCompanyName]’s bottom line.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to find another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf goes beyond single-site operations to accommodate multiple properties under one management company—organizing beverage cart requests, snack-bar orders, and to-go pickups efficiently across each facility.

We’re inviting 2–3 management companies to join us at no cost for 2025, ensuring our platform aligns perfectly with multi-course challenges. For instance, at Pinetree Country Club, we decreased average order times by 40%, helping keep players happy and reducing slowdowns.

Interested in a brief conversation about how this might scale for [ManagementCompanyName]? We’d love to share how Swoop can uplift all your properties.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now supports multi-facility operations—handling beverage cart deliveries, snack-bar orders, and to-go requests in a single interface. This way, you can standardize best practices across all properties managed by [ManagementCompanyName].

We’re selecting 2–3 management companies to partner with at no cost for 2025, ensuring our solution tackles real-world challenges. At Pinetree Country Club, our platform cut average order times by 40%, leading to satisfied players and seamless rounds.

I’d love to see if this resonates with [ManagementCompanyName]’s overall strategy. Let me know if you’re up for a short call.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf supports management companies by providing a unified on-course solution—coordinating beverage cart services, snack-bar orders, and to-go pickups across all properties in your portfolio. Our aim is to maximize efficiency and maintain a smooth pace of play, site by site.

We’re offering a no-cost partnership to 2–3 management companies in 2025, so we can align our platform to your real operational needs. For example, at Pinetree Country Club, we reduced average order times by 40%, boosting both revenue and guest satisfaction.

Would you be open to a quick call on how Swoop could assist [ManagementCompanyName] and its facilities? I’d be happy to walk you through our approach.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/management_companies"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
