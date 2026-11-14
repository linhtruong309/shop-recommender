-- Thuộc tính sản phẩm theo thời gian (đa số giá trị đã bị hash, trừ categoryid và available)
select
    to_timestamp_ntz(timestamp, 3)  as valid_from,
    itemid                          as item_id,
    lower(property)                 as property,
    value
from {{ source('raw', 'item_properties') }}
