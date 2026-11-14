-- Item features: category, parent category, train popularity
with items as (
    select distinct item_id from {{ ref('stg_events') }}
),

popularity as (
    select
        item_id,
        count(distinct visitor_id)  as n_train_visitors,
        sum(interaction_score)      as train_score
    from {{ ref('train_interactions') }}
    group by 1
)

select
    i.item_id,
    c.category_id,
    ct.parent_category_id,
    coalesce(p.n_train_visitors, 0)  as n_train_visitors,
    coalesce(p.train_score, 0)       as train_score
from items i
left join {{ ref('int_item_current_category') }} c using (item_id)
left join {{ ref('stg_category_tree') }} ct on ct.category_id = c.category_id
left join popularity p using (item_id)
