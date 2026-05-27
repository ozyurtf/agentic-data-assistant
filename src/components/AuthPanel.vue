<template>
    <div class="auth-wrapper">
        <div class="auth-card">
            <div class="auth-header">
                <h2 class="auth-title">Sign up or sign in to continue</h2>
            </div>

            <form class="auth-form" @submit.prevent="login">
                <label class="auth-field">
                    <span class="auth-field-label">Username</span>
                    <div class="auth-input-wrap">
                        <i class="fas fa-user auth-input-icon"></i>
                        <input
                            v-model="userId"
                            type="text"
                            autocomplete="username"
                            required
                            placeholder="Enter your username"
                            class="auth-input"
                        />
                    </div>
                </label>

                <label class="auth-field">
                    <span class="auth-field-label">Password</span>
                    <div class="auth-input-wrap">
                        <i class="fas fa-lock auth-input-icon"></i>
                        <input
                            v-model="password"
                            type="password"
                            autocomplete="current-password"
                            required
                            placeholder="Enter your password"
                            class="auth-input"
                        />
                    </div>
                </label>

                <p v-if="error" class="auth-error">
                    <i class="fas fa-exclamation-circle"></i> {{ error }}
                </p>

                <div class="auth-actions">
                    <button
                        type="button"
                        class="auth-btn auth-btn-secondary"
                        :disabled="busy"
                        @click="signup"
                    >
                        <span v-if="busyAction === 'signup'"><i class="fas fa-spinner fa-spin"></i></span>
                        <span v-else>Sign Up</span>
                    </button>
                    <button
                        type="submit"
                        class="auth-btn auth-btn-primary"
                        :disabled="busy"
                    >
                        <span v-if="busyAction === 'login'"><i class="fas fa-spinner fa-spin"></i></span>
                        <span v-else>Log In</span>
                    </button>
                </div>
            </form>
        </div>
    </div>
</template>

<script>
import { API_BASE_URL } from '@/config.js'

export default {
    name: 'AuthPanel',
    data () {
        return {
            userId: '',
            password: '',
            busyAction: '',
            error: ''
        }
    },
    computed: {
        busy () {
            return this.busyAction !== ''
        }
    },
    methods: {
        async post (path, action) {
            this.busyAction = action
            this.error = ''
            try {
                const response = await fetch(`${API_BASE_URL}${path}`, {
                    method: 'POST',
                    credentials: 'include',
                    headers: { 'Content-Type': 'application/json' },
                    // eslint-disable-next-line camelcase
                    body: JSON.stringify({ user_id: this.userId, password: this.password })
                })
                if (!response.ok) {
                    const detail = await response.json().catch(() => ({}))
                    throw new Error(detail.detail || `Request failed (${response.status})`)
                }
                const data = await response.json()
                this.$emit('authenticated', data.user_id)
            } catch (err) {
                this.error = err.message
            } finally {
                this.busyAction = ''
            }
        },
        signup () {
            if (!this.userId || !this.password) {
                this.error = 'Username and password are required'
                return
            }
            this.post('/api/signup', 'signup')
        },
        login () {
            if (!this.userId || !this.password) {
                this.error = 'Username and password are required'
                return
            }
            this.post('/api/login', 'login')
        }
    }
}
</script>

<style scoped>
.auth-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100%;
    padding: 24px 16px;
    background-color: #202020;
}

.auth-card {
    width: 100%;
    max-width: 360px;
    padding: 28px 24px;
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    color: #e8eaed;
}

.auth-header {
    text-align: center;
    margin-bottom: 22px;
}

.auth-logo {
    width: 52px;
    height: 52px;
    margin: 0 auto 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: linear-gradient(135deg, #4a90e2, #357ab7);
    box-shadow: 0 6px 18px rgba(74, 144, 226, 0.35);
}

.auth-logo i {
    color: #ffffff;
    font-size: 22px;
}

.auth-title {
    margin: 0;
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 0.2px;
    color: #cfd5e0;
}

.auth-form {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.auth-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.auth-field-label {
    font-size: 12.5px;
    font-weight: 500;
    letter-spacing: 0.2px;
    color: #8b94a3;
}

.auth-input-wrap {
    position: relative;
    display: flex;
    align-items: center;
}

.auth-input-icon {
    position: absolute;
    left: 12px;
    color: #6c7589;
    font-size: 13px;
    pointer-events: none;
}

.auth-input {
    width: 100%;
    padding: 10px 12px 10px 34px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    background-color: rgba(255, 255, 255, 0.04);
    color: #ffffff;
    font-size: 14px;
    transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;
}

.auth-input::placeholder {
    color: #6c7589;
}

.auth-input:focus {
    outline: none;
    border-color: #4a90e2;
    background-color: rgba(255, 255, 255, 0.06);
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.18);
}

.auth-error {
    margin: 0;
    padding: 8px 10px;
    border-radius: 6px;
    background-color: rgba(255, 107, 107, 0.12);
    color: #ff8a8a;
    font-size: 12.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.auth-actions {
    display: flex;
    gap: 10px;
    margin-top: 4px;
}

.auth-btn {
    flex: 1;
    padding: 11px 12px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.2px;
    cursor: pointer;
    transition: transform 0.08s ease, background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}

.auth-btn:active {
    transform: translateY(1px);
}

.auth-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.auth-btn-primary,
.auth-btn-secondary {
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: #d6dbe5;
}

.auth-btn-primary:hover:not(:disabled),
.auth-btn-secondary:hover:not(:disabled) {
    border-color: hsl(340 92% 52% / 1);
    background-color: hsl(340 92% 52% / 1);
    color: #ffffff;
    box-shadow: 0 6px 18px hsl(340 92% 52% / 0.4);
}
</style>
