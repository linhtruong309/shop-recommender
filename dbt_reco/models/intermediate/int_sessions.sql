-- Sessionize events: a gap longer than N minutes starts a new session
with ordered as (
    select
        *,
        lag(event_at) over (partition by visitor_id order by event_at) as prev_event_at
    from {{ ref('stg_events') }}
),

flagged as (
    select
        *,
        case
            when prev_event_at is null
              or datediff('minute', prev_event_at, event_at) > {{ var('session_timeout_minutes') }}
            then 1 else 0
        end as is_new_session
    from ordered
)

select
    visitor_id,
    item_id,
    event_type,
    event_at,
    transaction_id,
    visitor_id || '-' || sum(is_new_session) over (
        partition by visitor_id order by event_at
        rows between unbounded preceding and current row
    ) as session_id
from flagged
