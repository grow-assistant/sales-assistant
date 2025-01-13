import os

def main():
    # Define the email templates for an executive course without a full kitchen.
    # We keep references to “Pinetree Country Club” and focus on how Swoop Golf
    # can streamline limited F&B offerings (grab-and-go, snack bar, beverage cart).
    
    templates = {
        "fallback_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is a streamlined ordering solution that handles beverage cart requests, simple snack-bar orders, and to-go pickups. We understand that with no full kitchen on-site, quick and convenient service is essential—so golfers can grab what they need and keep playing.

We’re inviting 2–3 executive courses to join us at no cost for 2025, ensuring our platform meets the unique needs of smaller operations. For instance, at Pinetree Country Club, average order times dropped by 40%, leading to happier golfers and less congestion.

Interested in a short chat about how Swoop might benefit [FacilityName]? We’d love to show you how we can make a limited-menu setup work smoothly for everyone.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved beyond a single-service solution into a tailored platform—covering beverage cart deliveries, simple snack-bar orders, and efficient to-go pickups. It’s designed to boost your F&B potential, even if you don’t have a full kitchen on-site.

At Pinetree Country Club, we managed to reduce average order times by 40%, which kept the flow of play steady and guests satisfied. Do you have a few minutes to see if [FacilityName] could also benefit from this streamlined approach?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fallback_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf accommodates courses with limited food-service setups—managing beverage cart requests, grab-and-go snack orders, and quick to-go services. By simplifying the process, you can still offer a top-notch experience without the demands of a full kitchen.

At Pinetree Country Club, our platform lowered average order times by 40%, helping maintain a smooth pace. I’d love to discuss how Swoop could adapt to [FacilityName]’s specific constraints. Would a 10-minute call next week work for you?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf can act as a complete ordering hub—covering beverage cart requests, quick snack-bar items, and to-go orders. We understand an executive course may only have basic food prep or self-service options, which is why our streamlined platform is ideal for smaller menus.

We’re inviting 2–3 courses like yours to partner with us at no cost in 2025. One of our partners, Pinetree Country Club, experienced a 54% boost in F&B revenue simply by making orders more accessible throughout the round.

If you’re open to a quick conversation, I’d love to see if [FacilityName] could achieve a similar outcome. Feel free to let me know a good time to connect.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf is built to handle every F&B touchpoint—whether golfers want a beverage cart visit or a quick snack to go. By consolidating orders, we make it easier to run a limited-service operation, helping you maximize sales without the overhead of a full kitchen.

At Pinetree Country Club, average order times dropped by 40%. Let’s schedule a brief call to explore how Swoop could fit [FacilityName]’s setup. When might you have a few minutes next week?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "fb_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf can help [FacilityName] deliver a faster, more convenient experience—even if you don’t offer a full-service kitchen. We’ll handle on-course orders, quick snacks, and to-go pickups in a single platform, reducing the strain on your limited F&B resources.

At Pinetree Country Club, our approach led to a 54% jump in F&B revenue—a result we believe could translate to [FacilityName]. Would a 10-minute call on Thursday at 2 PM or Friday at 10 AM work to talk through the details? If another time is better, let me know.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_1.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to share how Swoop Golf’s ordering platform simplifies operations at executive courses that lack a full kitchen. With beverage cart deliveries, snack-bar items, and minimal prep needs, we streamline F&B so you can focus on a smooth guest experience.

We’re inviting 2–3 operations to join us at no cost for 2025. At Pinetree Country Club, we boosted F&B revenue by 54% and cut wait times by 40%. We think [FacilityName] could see similar gains, even with limited kitchen capabilities.

Would a 10-minute call on Thursday at 2 PM or Friday at 10 AM work to explore this further? If not, feel free to suggest another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_2.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf’s platform can help [FacilityName] deliver faster, more efficient service—whether that’s a quick grab-and-go snack or a beverage cart order. Our solution consolidates all F&B requests, easing the strain on staff in an environment without a full kitchen.

One facility we worked with saw a 54% uptick in F&B revenue and a 40% drop in wait times—results we believe are possible here as well.

Let’s set up a quick 10-minute call to see if our platform aligns with your goals. How does next Wednesday at 11 AM sound?

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "general_manager_initial_outreach_3.md": """Hey [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], I wanted to show you how Swoop Golf’s simplified platform can elevate the experience at [FacilityName]—even with minimal food-prep capabilities. By coordinating beverage cart deliveries, light snack options, and to-go items, we keep golfers satisfied without overextending your staff.

At Pinetree Country Club, our solution resulted in a 54% F&B revenue spike and a 40% reduction in wait times. Industry reports show that digital ordering can boost ancillary revenue by 20–40%. It could make a meaningful difference to your bottom line—even if your offerings are limited.

Would a quick 10-minute call on Thursday at 2 PM or Friday at 10 AM work to discuss? If not, I’m happy to find another time.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_1.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has been adapted for executive courses without a full kitchen—organizing beverage cart orders, snack-bar requests, and to-go pickups all in one place. This helps you maintain a smooth pace of play while still offering essential F&B options.

We’re inviting 2–3 facilities to join us at no cost for 2025, ensuring we align our platform to your specific needs. At Pinetree Country Club, this approach dropped average order times by 40%, keeping golfers happy and on schedule.

Interested in a quick chat on how this might work for [FacilityName]? We’d love to share how Swoop can make limited F&B options work effectively.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_2.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf now supports smaller-scale F&B setups, perfect for executive courses with limited food-prep capabilities. We centralize your snack-bar or beverage cart requests so golfers can stay focused on their round, and you can maintain efficiency.

We’re inviting 2–3 executive courses to partner with us at no cost for 2025, refining our solution to your real challenges. At Pinetree Country Club, average order times fell by 40%, which translated into more satisfied players overall.

Would you be up for a quick call to see if this fits [FacilityName]? Let me know, and I’ll share details on how Swoop can boost your operations.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
""",
        "golf_ops_initial_outreach_3.md": """Hi [FirstName],

[ICEBREAKER]

[SEASON_VARIATION], Swoop Golf has evolved to address the specific needs of executive courses—coordinating basic F&B services like beverage carts and grab-and-go snacks without the complexities of a full kitchen. Our mission is to keep play flowing and your guests satisfied.

We’re extending a no-cost partnership to 2–3 courses in 2025, ensuring we customize our platform to smaller menus. At Pinetree Country Club, we managed to reduce average order times by 40%, which significantly improved the golfer experience.

Ready to discuss how Swoop might work at [FacilityName]? I’d be happy to walk you through our approach in a short call.

Cheers,
Ty

Swoop Golf
480-225-9702
swoopgolf.com
"""
    }

    # Create the output directory
    output_dir = "docs/templates/executive_course_no_kitchen"
    os.makedirs(output_dir, exist_ok=True)

    # Write each template to its own .md file
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
