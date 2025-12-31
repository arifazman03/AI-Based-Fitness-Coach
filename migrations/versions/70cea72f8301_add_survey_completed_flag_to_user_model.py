"""Add survey_completed flag to User model

Revision ID: 70cea72f8301
Revises: d486df5d72b0
Create Date: 2025-11-17 14:40:24.234845

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import expression


# revision identifiers, used by Alembic.
revision = '70cea72f8301'
down_revision = 'd486df5d72b0'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add the column allowing NULLs temporarily, but with a default for new users
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('survey_completed', sa.Boolean(), 
                                      server_default=expression.false(), # Still useful for new rows
                                      nullable=True)) # <--- Set to TRUE temporarily!

    # 2. **CRITICAL:** Use SQL to explicitly set the column value to FALSE (0) for all EXISTING users
    op.execute("UPDATE users SET survey_completed = 0 WHERE survey_completed IS NULL")

    # 3. Alter the column to enforce NOT NULL constraint
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('survey_completed',
                              existing_type=sa.Boolean(),
                              nullable=False,
                              server_default=None) # Remove server default now that rows are populated

def downgrade():
    # Ensure your downgrade removes the column cleanly
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('survey_completed')

    # ### end Alembic commands ###
