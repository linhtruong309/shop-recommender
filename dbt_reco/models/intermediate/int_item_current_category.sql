-- Latest category per item (categoryid can change over time)
select
    item_id,
    try_to_number(value)  as category_id,
    valid_from            as category_valid_from
from {{ ref('stg_item_properties') }}
where property = 'categoryid'
qualify row_number() over (partition by item_id order by valid_from desc) = 1
