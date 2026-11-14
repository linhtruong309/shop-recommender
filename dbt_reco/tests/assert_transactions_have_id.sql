-- Sự kiện 'transaction' phải có transaction_id
select *
from {{ ref('stg_events') }}
where event_type = 'transaction' and transaction_id is null
