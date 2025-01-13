import os

def main():
    # Define the email templates for an country club,
    # preserving references to “Pinetree Country Club” and highlighting
    # top-tier amenities like pools, tennis courts, and fine dining.
    
    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has elevated its offering into a full-service luxury concierge platform—managing poolside F&B, tennis-court deliveries, on-course orders, and sophisticated to-go services. Our mission is to uphold the elevated standards your members expect while boosting efficiency.

We’re inviting 2–3 clubs to join us at no cost for 2025, to ensure we perfectly address the needs of top-tier properties. At Pinetree Country Club, this model reduced average order times by 40%, keeping members delighted and pace of play consistent.

Interested in a brief chat about how this might work for [ClubName]? We’d love to share how Swoop can enhance your members’ experience and preserve the exclusivity they value.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond traditional on-course service into a comprehensive club concierge—covering poolside and tennis-court F&B, on-course deliveries, and seamless to-go orders. We know how crucial it is to provide a refined, uninterrupted experience for members.

At Pinetree Country Club, we helped reduce average order times by 40%, ensuring that members remained satisfied and rounds weren’t disrupted. Would you have time for a quick call to see if [ClubName] could enjoy a similar impact?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf now acts as a luxury-level concierge platform—enhancing member satisfaction from the pool to the tennis courts, on-course deliveries, and fine-dining to-go services. We built it to maintain the exclusivity your members expect while minimizing operational strain.

At Pinetree Country Club, our approach lowered average order times by 40%, creating a smoother overall experience. I’d love to discuss how Swoop could meet the heightened expectations at [ClubName]. Would you be open to a 10-minute call next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has developed into a full-club luxury concierge platform—covering on-course ordering, poolside F&B, tennis-court deliveries, and refined to-go operations. We designed it to help you and your team deliver a seamless, exclusive experience to every member.

We're inviting 2–3 clubs to partner with us in 2025 at no cost. One of our current partners, Pinetree Country Club, saw a 54% surge in F&B revenue by making premium orders effortless for members, anytime, anywhere.

If you’re open to a quick chat, I'd love to see how Swoop could support [ClubName] and its exacting standards. Let me know a good time, and I can share references or more details tailored to your needs.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform delivers a white-glove service experience—covering poolside and tennis-court requests, on-course F&B, and flawlessly managed to-go orders. By centralizing everything, we help maintain a distinguished atmosphere while boosting operational efficiency.

One of our partners, Pinetree Country Club, saw a 40% decrease in average order times, supporting a more polished member experience. Let’s schedule a short call to explore how Swoop could meet [ClubName]’s high-end F&B needs. What does your availability look like next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is poised to help [ClubName] offer an impeccable member experience—from on-course orders to poolside and tennis-court service. By unifying multiple service points into one seamless system, we allow your team to focus on refined hospitality rather than juggling operations.

At Pinetree Country Club, our platform drove a 54% boost in F&B revenue—results we believe can be replicated at [ClubName]. Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore the possibilities? If another time is better, feel free to let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s luxury concierge platform can streamline [ClubName]’s operations. From tennis-court F&B delivery to poolside orders and to-go pickups, our solution caters to members’ every preference while maintaining the exclusivity your club is known for.

We're inviting 2–3 clubs to collaborate with us at no cost for 2025, customizing our approach to fit their unique needs. For example, at Pinetree Country Club, we increased F&B revenue by 54% and lowered wait times by 40%—results we believe [ClubName] could mirror.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss the potential? If another time is better, just let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf’s high-end platform to help [ClubName] maintain impeccable service—whether at the tee box, tennis court, or poolside. Our solution consolidates orders, reduces staff burdens, and upholds the elevated experience your members expect.

We recently worked with a club that realized a 54% boost in F&B revenue and a 40% cut in wait times—results we believe are feasible for [ClubName] as well.

Let’s book a brief 10-minute chat to see if our platform aligns with your goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to show how Swoop Golf’s luxury concierge platform can elevate [ClubName]’s member experience—from tennis-court and poolside orders to on-course service. By integrating order flow, we help reduce wait times and fuel F&B revenue growth.

Consider Pinetree Country Club: after adopting our platform, they experienced a 54% rise in F&B sales and a 40% drop in wait times. Data from top industry sources indicates that exclusive clubs using digital solutions can see a 20–40% increase in ancillary revenue—improving the bottom line without compromising the high standards members expect.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to find a more convenient time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a standard golf service into a luxury concierge platform—managing tennis-court F&B, poolside orders, and on-course deliveries. Our priority is to keep operations seamless so members can enjoy every aspect of your facility.

We’re inviting 2–3 clubs to join us at no cost for 2025, ensuring our platform perfectly addresses your unique standards. For example, at Pinetree Country Club, this approach shaved 40% off average order times, keeping members delighted and pace of play intact.

Interested in a quick conversation about how Swoop might work for [ClubName]? We’d love to share how our platform can enhance the member experience.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is now a high-end concierge platform—covering tennis-court deliveries, poolside F&B, and on-course requests. We aim to deliver a truly elite experience while preserving pace of play and reducing logistical headaches for your operations team.

We’re inviting 2–3 clubs to join us at no cost for 2025 to ensure our system meets every detail of your refined environment. At Pinetree Country Club, we minimized average order times by 40%, resulting in top-tier member satisfaction.

Interested in a short chat about how this might work for [ClubName]? We’d love to show you how Swoop can complement your exclusive setting.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf offers a fully integrated platform for clubs—managing on-course F&B requests, tennis-court deliveries, and poolside orders from one streamlined system. Our goal is to preserve a world-class member experience while ensuring operations run smoothly behind the scenes.

We’re extending a no-cost opportunity to 2–3 clubs in 2025, enabling us to tailor our platform to your specific standards. For instance, at Pinetree Country Club, we reduced average order times by 40%, leading to a more enjoyable, uninterrupted day for members.

Would a quick call on how this might fit [ClubName]’s vision make sense? I’d be happy to share how Swoop can enhance your already distinguished atmosphere.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/country_clubs"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
