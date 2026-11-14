-- Visitor-item interactions after the cutoff
with test as (
    select
        visitor_id,
        item_id,
        {{ interaction_metrics() }}
    from {{ ref('int_events_split') }}
    where split = 'test'
    group by 1, 2
),

known_visitors as (
    select distinct visitor_id from {{ ref('train_interactions') }}
)

select
    t.*,
    k.visitor_id is not null as is_known_visitor
from test t
left join known_visitors k using (visitor_id)
