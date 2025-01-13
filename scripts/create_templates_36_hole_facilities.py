import os

def main():
    # Define the email templates for multi-course (36+ holes) facilities.
    # We'll keep references to “Pinetree Country Club” for the success story
    # and focus on how Swoop Golf can benefit larger operations.
    
    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a simple golf service into a platform that can handle F&B requests across all 36+ holes at your facility—covering on-course deliveries, snack-bar orders, and to-go pickups. Our mission is to keep multiple courses running smoothly without sacrificing speed of play or guest satisfaction.

We’re inviting 2–3 large facilities to join us at no cost for 2025, ensuring our platform fully addresses multi-course operations. For instance, at Pinetree Country Club, we reduced average order times by 40%, boosting overall satisfaction and easing congestion.

Interested in a short call about how this could work for [FacilityName]? We’d love to show you how Swoop can help streamline day-to-day operations across both (or all) of your courses.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is now a comprehensive ordering platform—covering beverage cart deliveries, snack-bar requests, and convenient to-go pickups across all of your 36+ holes. We designed this solution to keep large facilities efficient, boost F&B revenue, and maintain a steady pace of play.

At Pinetree Country Club, average order times dropped by 40%, which helped prevent logjams on busy days. Would you have time for a quick call to see if [FacilityName] could benefit in a similar way?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf now operates as a platform built for facilities with multiple 18-hole courses—managing everything from beverage cart orders to snack-bar pickups. We aim to keep your golfers spread out across 36+ holes moving along without delays or confusion.

At Pinetree Country Club, our approach cut average order times by 40%, helping ensure a smoother round for everyone. I’d love to discuss how Swoop could help [FacilityName] maintain high service standards across all your courses. Would you be up for a 10-minute call next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf isn’t just for small operations—it’s built to handle F&B orders across multiple 18-hole layouts. Whether golfers are on the front or back side of either course, they can place orders seamlessly, and you can fulfill them efficiently without juggling separate systems.

We’re looking for 2–3 large facilities to partner with in 2025 at no cost. At Pinetree Country Club, for example, we helped drive a 54% boost in F&B revenue by making ordering easy at every turn.

If you’re open to a quick chat, I’d love to explore how [FacilityName] can see similar results. Let me know a good time to connect.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform is designed to handle the unique challenges of a 36+ hole facility—from coordinating beverage carts across multiple courses to organizing snack-bar orders and to-go pickups. By consolidating everything, we help you reduce operational bottlenecks, even when player traffic is split across two or more layouts.

One of our partners, Pinetree Country Club, saw a 40% decrease in average order times. Let’s schedule a short call to explore how Swoop could elevate F&B operations at [FacilityName]. What does your availability look like next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is ready to help [FacilityName] manage F&B across all 36 (or more) holes efficiently—from front-9 beverage cart requests to quick snack-bar pickups between the 18s. By uniting multiple service points into one system, we simplify operations and elevate the player experience.

At Pinetree Country Club, we drove a 54% boost in F&B revenue—a success we believe can be replicated at larger facilities like yours. Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss further? If another time suits better, just let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course ordering platform can streamline [FacilityName]’s operations across your multiple 18-hole courses. We centralize beverage cart deliveries, snack-bar orders, and to-go pickups, reducing staff strain even when both (or all) courses are busy.

We’re inviting 2–3 large facilities to partner with us at no cost for 2025. At Pinetree Country Club, we lifted F&B revenue by 54% and decreased wait times by 40%. We think [FacilityName] could see comparable gains.

Would a 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If not, feel free to suggest another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf can help [FacilityName] deliver faster, more efficient service—whether players are on Course A or Course B. Our all-in-one solution consolidates orders and reduces staff workload, enhancing the experience for golfers playing any (or all) of your 36+ holes.

We recently worked with a facility that achieved a 54% uptick in F&B revenue and a 40% cut in wait times—an improvement we believe is reproducible at [FacilityName].

Let’s set up a quick 10-minute conversation to see if our platform aligns with your needs. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d like to show how Swoop Golf’s platform can elevate the experience at [FacilityName]—offering on-course deliveries and snack-bar ordering across your entire 36-hole complex. By unifying order flow, we help reduce wait times and boost overall F&B revenue.

At Pinetree Country Club, the addition of our solution led to a 54% increase in F&B sales and a 40% reduction in wait times. Industry data suggests digital ordering can bring a 20–40% jump in ancillary revenue, a substantial gain for multi-course facilities like yours.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss further? If not, I’m happy to find another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is a full on-course ordering platform designed for multi-course facilities—managing everything from beverage cart deliveries to snack-bar requests and to-go pickups across 36+ holes. Our priority is to maintain efficient, hassle-free play across all layouts.

We’re inviting 2–3 facilities to join us at no cost for 2025, ensuring we tailor the platform to your dual or triple-course setup. For example, at Pinetree Country Club, we cut average order times by 40%, improving the overall golfer experience and keeping rounds on schedule.

Interested in a quick chat on how this could work for [FacilityName]? We’d love to discuss how Swoop can support your multi-course operations.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf consolidates F&B requests—beverage cart deliveries, snack-bar orders, and to-go pickups—across large facilities. We know how crucial it is to keep play flowing, especially when you’re juggling 36 or more holes simultaneously.

We’re inviting 2–3 multi-course operations to join us at no cost for 2025, making sure we address the real challenges of hosting players on multiple tracks. At Pinetree Country Club, this approach dropped average order times by 40%, boosting satisfaction and maintaining pace of play.

Interested in a brief conversation about how this might help [FacilityName]? We’d love to walk you through the details.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now accommodates facilities with multiple courses—coordinating beverage carts, snack-bar orders, and to-go requests to keep your 36+ holes running without a hitch. Our goal is to minimize downtime and maximize convenience for players, regardless of which course they’re on.

We’re offering 2–3 large facilities a no-cost partnership for 2025, letting us fine-tune our platform to your specific needs. For example, at Pinetree Country Club, we reduced average order times by 40%, maintaining smooth rounds all day.

Would you be open to a quick chat about how Swoop might streamline operations at [FacilityName]? I’d be glad to share more details.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/36_hole_facilities"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
