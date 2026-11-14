-- Time-versioned item properties (values are hashed except categoryid and available)
select
    to_timestamp_ntz(timestamp, 3)  as valid_from,
    itemid                          as item_id,
    lower(property)                 as property,
    value
from {{ source('raw', 'item_properties') }}
