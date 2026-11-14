-- Visitor-item interactions before the cutoff
select
    visitor_id,
    item_id,
    {{ interaction_metrics() }}
from {{ ref('int_events_split') }}
where split = 'train'
group by 1, 2
