breakdown_items = []
    for i, (category, amount) in enumerate(split.items()):
        daily_amount = amount / days if days > 0 else 0
        per_person = amount / people if people > 0 else 0
        
        breakdown_items.append(f"""
        <div class="budget-breakdown-item">
            <div class="category-color" style="background:{colors[i % len(colors)]};"></div>
            <div class="category-name">{get_category_emoji(category)} {category}</div>
            <div class="category-amount">₹{amount:,}</div>
            <div class="category-daily">₹{daily_amount:,.0f}/day</div>
            <div class="category-per-person">₹{per_person:,.0f}/person</div>
            <div class="category-percentage">{percentages[i]:.1f}%</div>
        </div>
        """)