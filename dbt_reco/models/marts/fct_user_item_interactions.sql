-- Visitor-item interactions over the full period
select
    s.visitor_id,
    s.item_id,
    c.category_id,
    {{ interaction_metrics() }}
from {{ ref('int_sessions') }} s
left join {{ ref('int_item_current_category') }} c using (item_id)
group by 1, 2, 3
