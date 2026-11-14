-- Every 'transaction' event must have a transaction_id
select *
from {{ ref('stg_events') }}
where event_type = 'transaction' and transaction_id is null
