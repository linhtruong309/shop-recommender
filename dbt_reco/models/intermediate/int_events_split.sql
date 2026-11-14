-- Label each event as train or test: last N days = test
with bounds as (
    select dateadd('day', -{{ var('test_window_days') }}, max(event_at)) as cutoff_at
    from {{ ref('int_sessions') }}
)

select
    e.*,
    b.cutoff_at,
    case when e.event_at < b.cutoff_at then 'train' else 'test' end as split
from {{ ref('int_sessions') }} e
cross join bounds b
