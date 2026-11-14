{# Interaction counts and weighted score: view = 1, addtocart = 3, transaction = 5 #}
{% macro interaction_metrics() %}
    count_if(event_type = 'view')         as n_views,
    count_if(event_type = 'addtocart')    as n_addtocart,
    count_if(event_type = 'transaction')  as n_transactions,
    count(distinct session_id)            as n_sessions,
    n_views * 1 + n_addtocart * 3 + n_transactions * 5 as interaction_score,
    min(event_at)                         as first_interaction_at,
    max(event_at)                         as last_interaction_at
{% endmacro %}
