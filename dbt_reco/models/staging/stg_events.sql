-- One row per visitor event on an item
select
    to_timestamp_ntz(timestamp, 3)  as event_at,   -- epoch ms -> timestamp
    visitorid                       as visitor_id,
    itemid                          as item_id,
    lower(event)                    as event_type,
    transactionid                   as transaction_id
from {{ source('raw', 'events') }}
where visitorid is not null
  and itemid is not null
