use anchor_lang::prelude::*;
use anchor_spl::{
    token_2022::{self, Token2022},
    token_interface::{Mint, TokenAccount},
};

declare_id!("GRACEgNS1111111111111111111111111111111111"); // Replace with your deployed Program ID

#[program]
pub mod grace_token {
    use super::*;

    /// Initialize the GRACE token mint with Token-2022 and config PDA
    pub fn initialize(
        ctx: Context<Initialize>,
        name: String,
        symbol: String,
        uri: String,
        total_supply: u64,
        fee_basis_points: u16, // e.g. 50 = 0.5%
    ) -> Result<()> {
        // Initialize Token-2022 mint (full extensions in production)
        let mint = &ctx.accounts.mint;

        let cpi_accounts = token_2022::MintTo {
            mint: mint.to_account_info(),
            to: ctx.accounts.recipient.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        };
        let cpi_ctx = CpiContext::new(ctx.accounts.token_program.to_account_info(), cpi_accounts);
        token_2022::mint_to(cpi_ctx, total_supply)?;

        // Store config for fees/treasury
        let config = &mut ctx.accounts.config;
        config.treasury = ctx.accounts.treasury.key();
        config.fee_basis_points = fee_basis_points;
        config.bump = ctx.bumps.config;

        // TODO: Immediately revoke mint authority post-initialization for fairness
        // Use token_2022::set_authority to set authority to None

        Ok(())
    }

    /// Process protocol fee (hook into bonding curve or trade instructions)
    pub fn process_fee(ctx: Context<ProcessFee>, amount: u64) -> Result<()> {
        let fee = ((amount as u128) * (ctx.accounts.config.fee_basis_points as u128) / 10000) as u64;
        
        // Transfer SOL fee to treasury (DAO-controlled, later off-ramped compliantly)
        **ctx.accounts.payer.try_borrow_mut_lamports()? -= fee;
        **ctx.accounts.treasury.try_borrow_mut_lamports()? += fee;

        Ok(())
    }
}

#[account]
#[derive(Default)]
pub struct Config {
    pub treasury: Pubkey,
    pub fee_basis_points: u16,
    pub bump: u8,
}

#[derive(Accounts)]
#[instruction(name: String, symbol: String, uri: String)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        mint::decimals = 9,
        mint::authority = authority,
    )]
    pub mint: InterfaceAccount<'info, Mint>,

    #[account(
        init_if_needed,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = authority,
    )]
    pub recipient: InterfaceAccount<'info, TokenAccount>,

    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 2 + 1,
        seeds = [b"config"],
        bump
    )]
    pub config: Account<'info, Config>,

    #[account(mut)]
    pub authority: Signer<'info>,

    /// CHECK: Community multisig / DAO treasury wallet
    #[account(mut)]
    pub treasury: UncheckedAccount<'info>,

    pub token_program: Interface<'info, Token2022>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ProcessFee<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    /// CHECK: Treasury
    #[account(mut)]
    pub treasury: UncheckedAccount<'info>,
    #[account(mut)]
    pub config: Account<'info, Config>,
}