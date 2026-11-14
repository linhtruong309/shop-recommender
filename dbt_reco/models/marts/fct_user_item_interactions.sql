-- Bảng huấn luyện: mỗi cặp (visitor, item) với tín hiệu ẩn (implicit feedback)
-- Trọng số: view = 1, addtocart = 3, transaction = 5
select
    s.visitor_id,
    s.item_id,
    c.category_id,
    count_if(s.event_type = 'view')         as n_views,
    count_if(s.event_type = 'addtocart')    as n_addtocart,
    count_if(s.event_type = 'transaction')  as n_transactions,
    count(distinct s.session_id)            as n_sessions,
    n_views * 1 + n_addtocart * 3 + n_transactions * 5 as interaction_score,
    min(s.event_at)                         as first_interaction_at,
    max(s.event_at)                         as last_interaction_at
from {{ ref('int_sessions') }} s
left join {{ ref('int_item_current_category') }} c using (item_id)
group by 1, 2, 3
