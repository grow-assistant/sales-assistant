import os

def main():
    # Define templates for facilities with high daily fees.
    # Keep references to “Pinetree Country Club” as requested.
    # Remove references to pools, tennis courts, or membership.
    # Maintain placeholders like [FacilityName], [FirstName], [ICEBREAKER], [SEASON_VARIATION].

    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a golf-only service into a comprehensive on-course ordering solution—managing beverage cart requests, snack-bar orders, and to-go pickups. Our goal is to help you deliver a premium experience and maintain a steady pace of play.

We’re inviting 2–3 facilities to partner with us at no cost for 2025. For instance, at Pinetree Country Club, this approach helped reduce average order times by 40%, leading to happier players and minimized slowdowns.

Interested in a quick chat about how this might work for [FacilityName]? We’d love to share how Swoop can help elevate your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has grown from a simple golf service into a fully integrated platform—covering beverage cart coordination, snack-bar deliveries, and seamless to-go pickups. We aim to keep operations running smoothly while elevating the overall experience for golfers who expect the best.

At Pinetree Country Club, average order times dropped by 40%, resulting in happier golfers and less disruption on the course. Would you have time for a quick call to discuss if [FacilityName] could see similar improvements?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf now offers a comprehensive on-course solution—handling beverage cart requests, snack-bar orders, and to-go services. We’ve designed it to streamline high-volume operations and keep the pace of play on track for premium golfers.

At Pinetree Country Club, our platform helped reduce average order times by 40%, ensuring players stayed satisfied and focused on their round. I’d love to connect for a brief discussion on how Swoop could benefit [FacilityName]. Would you be open to a 10-minute call next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is more than a basic golf service—we’re an on-course ordering platform built to boost F&B revenue while delivering a top-tier experience. By centralizing beverage cart and snack-bar requests, we help you serve premium clientele quickly and efficiently.

We’re looking for 2–3 facilities to partner with at no cost in 2025. Pinetree Country Club, for example, saw a 54% boost in F&B revenue by making on-course orders simple and accessible.

If you’re open to a short call, I’d love to see if [FacilityName] could experience similar success. Let me know a good time to connect, and I can share references or more details tailored to your needs.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s on-course ordering platform has expanded to handle beverage cart requests, snack-bar deliveries, and quick to-go orders—perfect for facilities that cater to discerning golfers. By consolidating these services, we reduce bottlenecks and boost efficiency, while preserving a seamless guest experience.

At Pinetree Country Club, we helped drop average order times by 40%. Let’s schedule a short call to see how Swoop might enhance F&B operations at [FacilityName]. Any availability next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf can help [FacilityName] deliver a higher level of service—from on-course orders to convenient to-go pickups. By integrating these service points into one platform, you reduce wait times and streamline operations for guests who expect a premium experience.

At Pinetree Country Club, our platform drove a 54% boost in F&B revenue—outcomes we believe could apply at [FacilityName] too. Would a brief 10-minute call on Thursday at 2 PM or Friday at 10 AM work? If another time is better, feel free to let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course ordering platform could help [FacilityName] offer a seamless experience. From beverage cart deliveries to snack-bar orders and to-go pickups, we make it easy to provide top-shelf service without overburdening staff.

We’re inviting 2–3 facilities to partner with us at no cost for 2025, tailoring our platform to your unique needs. At Pinetree Country Club, we helped increase F&B revenue by 54% and reduce wait times by 40%—results we believe [FacilityName] could replicate.

Would a 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If neither time is ideal, just let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf’s platform to help [FacilityName] deliver quicker, more efficient on-course service—whether through beverage carts or snack-bar pickup. By consolidating orders, we reduce staff strain and enhance the golfer experience, which is crucial for a higher-priced round.

We recently worked with a facility that saw a 54% jump in F&B revenue and a 40% drop in wait times—a transformation we believe could happen at [FacilityName] as well.

Let’s set up a quick 10-minute conversation to see if our platform aligns with your goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s on-course platform can elevate the player experience at [FacilityName]. By streamlining beverage cart orders and snack-bar deliveries, we help reduce wait times and boost F&B revenue—key factors for facilities charging premium rates.

Consider Pinetree Country Club: after adopting our platform, they saw a 54% increase in F&B sales and a 40% reduction in wait times. According to industry data, using digital ordering often results in a 20–40% bump in ancillary revenue, which can significantly impact your bottom line.

Would a 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to arrange another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved from a basic golf service into a full on-course ordering platform—managing beverage cart deliveries, snack-bar requests, and to-go orders. Our mission is to help you offer a polished experience and keep the pace of play moving at a top-tier level.

We’re inviting 2–3 facilities to join us at no cost for 2025, to ensure the platform meets your high-end requirements. For instance, at Pinetree Country Club, we reduced average order times by 40%, leaving golfers happier and less prone to slow play.

Interested in a quick conversation about how this might work at [FacilityName]? We’d love to share how Swoop can enhance your guests’ experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now offers a unified solution for on-course F&B—covering beverage cart coordination, snack-bar orders, and to-go pickups. We designed it to keep your higher-paying golfers fully satisfied while maintaining a smooth pace of play.

We’re inviting 2–3 facilities to join us at no cost for 2025 to refine our platform further. For example, at Pinetree Country Club, this approach cut average order times by 40%, creating a more enjoyable round for everyone.

Interested in a short chat on how this might fit [FacilityName]? We’d love to walk you through our process.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has become a complete on-course solution—handling beverage cart requests, snack-bar orders, and efficient to-go pickups. Our goal is to streamline F&B while ensuring golfers who pay a premium feel well-served and stay on pace.

We’re inviting 2–3 facilities to partner with us in 2025 at no cost, so we can fine-tune our system around your needs. For example, Pinetree Country Club reported a 40% drop in average order times after implementing Swoop, leading to smoother rounds and happier guests.

Would you be open to a brief call about how this could work for [FacilityName]? I’d be happy to provide details and next steps.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/high_daily_fee"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
