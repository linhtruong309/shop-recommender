-- Train must end before test starts
select *
from (
    select
        (select max(last_interaction_at) from {{ ref('train_interactions') }})  as train_end,
        (select min(first_interaction_at) from {{ ref('test_interactions') }})  as test_start
)
where train_end >= test_start
