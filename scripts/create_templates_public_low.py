import os

def main():
    # Define the email templates for public facilities with lower daily fees.
    # Keeping references to “Pinetree Country Club” in the success story, as requested.
    # Avoiding direct references to memberships, pools, or tennis. 
    # Replacing “ClubName” with “FacilityName” to keep it broad.

    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a golf-only service into a user-friendly on-course ordering solution—managing beverage cart requests, snack-bar orders, and to-go pickups. Our goal is to optimize operations and keep golfers moving at a steady pace.

We’re inviting 2–3 facilities to join us at no cost for 2025, ensuring we meet the specific needs of those with a lower daily fee structure. For instance, at Pinetree Country Club, this approach helped reduce average order times by 40%, leading to happier players and fewer slowdowns.

Interested in a quick chat about how this could work for [FacilityName]? We’d love to show you how Swoop can enhance your guests’ experience and streamline your operation.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has grown from a simple golf service into a fully integrated on-course platform—handling beverage cart deliveries, snack-bar requests, and quick to-go pickups. We strive to help facilities serving cost-conscious golfers increase efficiency, boost F&B revenue, and maintain pace of play.

At Pinetree Country Club, we helped reduce average order times by 40%, which led to smoother rounds and more satisfied players. Would you have time for a quick call to see if [FacilityName] could see similar improvements?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf now operates as a complete on-course solution—covering beverage cart requests, snack-bar orders, and to-go pickups. We’ve designed it specifically to help facilities with lower green fees run more efficiently and keep golfers content.

At Pinetree Country Club, our platform lowered average order times by 40%, minimizing backups and keeping players engaged. I’d love to discuss how we could help [FacilityName] achieve similar results. Would a 10-minute call next week work for you?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is more than a golf service—it’s an on-course ordering platform that makes it easy for you and your team to serve guests quickly. Whether they’re requesting the beverage cart or grabbing a snack-bar order, we centralize everything so you don’t have to juggle multiple systems.

We’re looking for 2–3 facilities to partner with in 2025 at no cost. One of our current partners, Pinetree Country Club, saw a 54% boost in F&B revenue by streamlining orders and making it simple for players to purchase during their round.

If you’re open to a short chat, I’d love to see if [FacilityName] could benefit similarly. Let me know a good time to connect.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform has expanded to handle beverage cart requests, snack-bar deliveries, and to-go orders—all in one place. By consolidating these services, we help improve efficiency, reduce wait times, and keep golf rounds moving for those with tighter budgets.

Pinetree Country Club, for example, saw a 40% reduction in average order times. I’d love to schedule a short call to explore how Swoop could elevate operations at [FacilityName]. What’s your availability next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is ready to help [FacilityName] deliver a smoother F&B experience—from on-course orders to easy pickup options. By integrating these service points into one system, we reduce wait times and add convenience for players who value cost and efficiency.

At Pinetree Country Club, our approach led to a 54% boost in F&B revenue—a result we believe could translate to [FacilityName] as well. Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If another time is better, let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course ordering platform can simplify operations at [FacilityName]. From beverage cart requests to snack-bar orders and to-go pickups, everything is centralized to help lower-fee facilities maintain quality service without overburdening staff.

We’re inviting 2–3 operations to partner with us at no cost for 2025 to customize our platform for their specific needs. For example, at Pinetree Country Club, F&B revenue jumped 54% and wait times dropped 40%—results we think [FacilityName] could replicate.

Would a short call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If those times aren’t ideal, feel free to suggest another.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf’s on-course platform to help [FacilityName] deliver faster, more efficient service. By consolidating beverage cart and snack-bar orders, you can reduce staff stress while improving guest satisfaction—especially important if your rates are designed to attract a broad range of golfers.

We recently worked with a club that experienced a 54% bump in F&B revenue and a 40% drop in wait times—a transformation we believe [FacilityName] could also see.

Let’s set up a 10-minute chat to see if our solution aligns with your goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course system can enhance the experience at [FacilityName]. By streamlining orders and deliveries, we help facilities offering lower daily fees reduce wait times and increase F&B revenue.

Look at Pinetree Country Club: they saw a 54% increase in F&B sales and a 40% reduction in wait times after adopting our platform. Industry data shows that adopting digital ordering typically leads to a 20–40% boost in ancillary revenue—key for building profitability at any rate level.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to find another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved into a complete on-course ordering platform—coordinating beverage cart deliveries, snack-bar requests, and to-go orders. Our goal is to boost operational efficiency and keep pace of play steady, especially for those focused on affordability.

We’re inviting 2–3 facilities to join us at no cost for 2025, ensuring we align our platform to your needs. For example, at Pinetree Country Club, our approach cut average order times by 40%, resulting in smoother rounds and happier golfers.

Interested in a quick chat about how this might help [FacilityName]? We’d love to show you how Swoop can enrich your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now offers a one-stop solution for on-course F&B—covering beverage cart deliveries, snack-bar orders, and to-go pickups. We’re here to keep golfers active on the course while increasing F&B revenue for cost-focused facilities.

We’re inviting 2–3 operations to partner with us at no cost in 2025, fine-tuning our platform to your specific requirements. At Pinetree Country Club, average order times dropped 40%, keeping players engaged and on schedule.

Interested in a brief call to explore if this fits [FacilityName]? Let me know, and I’ll share how Swoop can uplift your day-to-day operations.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf provides a fully integrated approach to on-course service—taking care of beverage cart orders, snack-bar requests, and to-go pickups. We focus on improving operational flow so you can keep your golfers satisfied, even at a more budget-friendly price point.

We’re offering 2–3 facilities a no-cost partnership for 2025 to ensure our platform meets real-world needs. For instance, at Pinetree Country Club, we helped cut average order times by 40%, boosting overall golfer satisfaction.

Would you be open to a quick conversation on how this could work for [FacilityName]? I’d be happy to walk you through the details.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/public_clubs_low_fee"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
