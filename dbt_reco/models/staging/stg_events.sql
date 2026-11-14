-- Một dòng = một sự kiện của visitor trên một sản phẩm
select
    to_timestamp_ntz(timestamp, 3)  as event_at,   -- epoch ms -> timestamp
    visitorid                       as visitor_id,
    itemid                          as item_id,
    lower(event)                    as event_type,
    transactionid                   as transaction_id
from {{ source('raw', 'events') }}
where visitorid is not null
  and itemid is not null
