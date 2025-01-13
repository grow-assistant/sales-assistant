import os

def main():
    # Define the email templates, preserving references to "Pinetree Country Club"
    # and removing references to pools, tennis courts, or membership.
    # We also avoid using words like "public course" or "country club" (except for "Pinetree Country Club"),
    # focusing instead on language that resonates with daily-fee facilities.

    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a golf-only service into a streamlined on-course ordering solution—managing beverage cart requests, snack-bar orders, and to-go pickups. Our goal is to enhance your operation’s efficiency and keep the pace of play steady.

We’re inviting 2–3 facilities to join us at no cost for 2025, to ensure we’re truly meeting your needs. For instance, at Pinetree Country Club, this approach helped reduce average order times by 40%, keeping players happier and minimizing slowdowns.

Interested in a quick chat on how this might work for [FacilityName]? We’d love to share how Swoop can elevate your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has grown from a simple golf service into a fully integrated ordering platform—covering on-course deliveries, beverage cart coordination, and convenient to-go pickups. We aim to keep operations efficient, boost F&B revenue, and maintain a smooth pace of play.

At Pinetree Country Club, we helped reduce average order times by 40%, leading to happier golfers and less congestion. Would you have time for a quick chat to see if [FacilityName] could achieve similar results?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf now operates as a full-service on-course solution—handling beverage cart orders, snack-bar requests, and to-go services. We built this to increase efficiency and keep golfers moving at a steady pace.

At Pinetree Country Club, our approach helped lower average order times by 40%, ensuring players remain engaged in the game. I’d love to connect for a brief discussion on how Swoop could support [FacilityName] in similar ways. Would you be open to a 10-minute call next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf can act as a complete on-course ordering platform—covering beverage cart requests, snack-bar orders, and streamlined to-go operations. We designed it to help you and your team serve guests quickly and conveniently, without juggling multiple systems.

We're looking for 2–3 facilities to partner with in 2025 at no cost. One of our current partners, Pinetree Country Club, saw a 54% boost in F&B revenue by making orders simple and accessible for players on the course.

If you're open to a quick chat, I'd love to see if [FacilityName] could benefit in the same way. Feel free to let me know a good time to connect, and I can share references or more details tailored to your needs.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform has expanded into a complete on-course solution—covering beverage cart requests, snack-bar deliveries, and to-go orders. By centralizing these services, we help reduce bottlenecks, boost efficiency, and keep golfers on the move.

One of our recent partners, Pinetree Country Club, saw a 40% decrease in average order times. Let’s schedule a short call to explore how Swoop could enhance [FacilityName]’s operations. What does your availability look like next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is poised to help [FacilityName] improve guest satisfaction—whether it’s on-course orders or to-go pickups. By integrating multiple service points into one streamlined system, we help you reduce wait times and operational complexity.

At Pinetree Country Club, our platform drove a 54% boost in F&B revenue—a success we believe can be replicated at [FacilityName]. Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If another time is better, feel free to let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s new on-course platform could help streamline [FacilityName]’s operations. From beverage cart deliveries to snack-bar orders and to-go pickups, our platform makes it easy to provide seamless service without overstraining your staff.

We're inviting 2–3 facilities to partner with us at no cost for 2025 to tailor our platform to your unique needs. For example, at Pinetree Country Club, we helped increase F&B revenue by 54% and reduced wait times by 40%—results we believe [FacilityName] could replicate.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If another time is better, feel free to let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf’s platform to help [FacilityName] deliver faster, more efficient service—whether it’s via beverage cart or snack-bar orders. Our solution consolidates orders and reduces staff strain, ultimately driving a better guest experience and boosting profitability.

We recently worked with a facility that saw a 54% jump in F&B revenue and a 40% drop in wait times—a transformation we believe is replicable at [FacilityName].

Let’s set up a brief 10-minute conversation to see if our platform aligns with your goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course platform can elevate [FacilityName]’s guest experience—whether they’re ordering from a beverage cart or picking up snacks on the turn. By streamlining order flow, we help reduce wait times and boost F&B revenue.

Consider the success story of Pinetree Country Club: after implementing our platform, they saw a 54% increase in F&B sales and a 40% reduction in wait times. Golf Inc. reports that facilities using digital ordering platforms often experience a 20–40% increase in ancillary revenue, illustrating the profound impact this could have on your bottom line.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to find another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a golf-only service into a full on-course ordering platform—managing beverage cart deliveries, snack-bar requests, and to-go orders. Our goal is to enhance efficiency and keep rounds flowing smoothly without disrupting play.

We’re inviting 2–3 facilities to join us at no cost for 2025, to ensure we’re truly meeting your needs. For instance, at Pinetree Country Club, this approach helped reduce average order times by 40%, keeping players happier and minimizing slowdowns.

Interested in a quick chat on how this might work for [FacilityName]? We’d love to share how Swoop can elevate your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has expanded beyond a traditional golf service into a one-stop ordering platform for on-course F&B—covering beverage cart deliveries, snack-bar requests, and efficient to-go options. Our mission is to keep the pace of play uninterrupted while boosting your F&B revenue.

We’re inviting 2–3 facilities to join us at no cost for 2025, to ensure we’re truly meeting your needs. For instance, at Pinetree Country Club, this approach helped reduce average order times by 40%, keeping golfers satisfied and play on schedule.

Interested in a quick chat on how this might work for [FacilityName]? We’d love to discuss how Swoop can benefit your operation.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now offers a comprehensive on-course platform—handling beverage cart orders, snack-bar requests, and quick to-go pickups. Our goal is to streamline F&B operations and maintain a steady pace of play.

We’re inviting 2–3 facilities to join us at no cost for 2025, to ensure we’re truly meeting your needs. For instance, at Pinetree Country Club, we helped reduce average order times by 40%, keeping players happier and on schedule.

Interested in a quick chat on how this might work for [FacilityName]? We’d love to share how Swoop can elevate your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory (if it doesn't already exist)
    output_dir = "docs/templates/facility"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to a separate .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
